import os
import tempfile
import subprocess
import torch
from flask import Flask, request, render_template, jsonify, Response
from werkzeug.utils import secure_filename
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sentencepiece as spm
import onmt
import onmt.opts as opts
import onmt.translate.translator as translator
from onmt.translate.beam_search import BeamSearch
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser
from datetime import datetime
import librosa
import numpy as np
import pyaudio
import whisper
import threading
import queue
import json
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path configurations
WHISPER_MODEL_PATH = "openai/whisper-small"  
WHISPER_WEIGHTS_PATH = "model/whisper_pretrained_converted_weights.pth"  
ONMT_MODEL_PATH = "model/model.fren_step_30000.pt"  
SOURCE_SP_MODEL = "model/source.model" 
TARGET_SP_MODEL = "model/target.model"  

# Global variables to store loaded models
asr_processor = None
asr_model = None
sp_source = None
sp_target = None

# Global variables for real-time streaming
audio_queue = queue.Queue()
stream_active = False
p = None
stream = None
chunk_size = 4096  # Increased chunk size for better processing
buffer_duration = 3  # Process 3 seconds of audio at a time
input_device_index = None  # Will be set dynamically

# Load the models
def load_asr_model():
    print("Loading ASR model...")
    # Load the base Whisper model
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
    
    # Load fine-tuned weights 
    if os.path.exists(WHISPER_WEIGHTS_PATH):
        fine_tuned_weights = torch.load(WHISPER_WEIGHTS_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model_dict = model.state_dict()
        for key, value in fine_tuned_weights.items():
            if key in model_dict:
                model.state_dict()[key].copy_(value)
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return processor, model

def load_translation_model():
    print("Loading translation model...")
    
    # Load SentencePiece models
    sp_source = spm.SentencePieceProcessor()
    sp_target = spm.SentencePieceProcessor()
    sp_source.load(SOURCE_SP_MODEL)
    sp_target.load(TARGET_SP_MODEL)
    
    return None, sp_source, sp_target

def list_audio_devices():
    """List all available audio input devices"""
    try:
        p_temp = pyaudio.PyAudio()
        info = []
        
        for i in range(p_temp.get_device_count()):
            device_info = p_temp.get_device_info_by_index(i)
            # Only include devices with input channels
            if device_info.get('maxInputChannels', 0) > 0:
                info.append({
                    'index': i,
                    'name': device_info.get('name', 'Unknown Device'),
                    'defaultSampleRate': device_info.get('defaultSampleRate', 44100),
                    'maxInputChannels': device_info.get('maxInputChannels', 0)
                })
                print(f"Device {i}: {device_info.get('name')} (Inputs: {device_info.get('maxInputChannels')})")
        
        p_temp.terminate()
        return info
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        return []

def transcribe_audio(processor, model, audio_array, sampling_rate=16000):
    """Transcribe audio using the Whisper model (Tagalog input)"""
    try:
        # Debug info about the audio
        duration = len(audio_array) / sampling_rate
        print(f"Audio duration: {duration:.2f} seconds, samples: {len(audio_array)}, sampling rate: {sampling_rate}")
        
        # Skip very short audio (likely noise or silence)
        if duration < 0.5:  # Less than 500ms is likely not speech
            print(f"Audio too short ({duration:.2f}s), skipping transcription")
            return ""
            
        # Process audio with the model
        input_features = processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        input_features = input_features.to(model.device)
        
        # Generate transcription (explicitly setting language to Tagalog)
        with torch.no_grad():
            # Force Tagalog language detection and transcribe task
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="tl", task="transcribe")
            print(f"Using forced_decoder_ids for Tagalog: {forced_decoder_ids}")
            
            generated_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448  # Longer max length for better context
            )
        
        # Decode the generated IDs
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Raw transcription result: '{transcription}'")
        
        return transcription
        
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        import traceback
        print(traceback.format_exc())
        return ""

def translate_text(sp_source, sp_target, text):
    """Translate text from Tagalog to English using OpenNMT-py"""
    if not text or len(text.strip()) == 0:
        return ""
        
    # Tokenize the source text
    tokenized_text = sp_source.encode(text, out_type=str)
    tokenized_line = " ".join(tokenized_text)
    
    # Create a temporary source file with the tokenized text
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_src:
        temp_src.write(tokenized_line)
        temp_src_path = temp_src.name
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_out:
        temp_out_path = temp_out.name
    
    try:
        # Use the OpenNMT CLI directly
        cmd = [
            'python', '-m', 'onmt.bin.translate',
            '-model', ONMT_MODEL_PATH,
            '-src', temp_src_path,
            '-output', temp_out_path,
            '-gpu', '0' if torch.cuda.is_available() else '-1',
            '-verbose'
        ]
        
        print(f"Running translation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Translation command output: {result.stdout}")
        
        # Read the translation
        with open(temp_out_path, 'r') as f:
            translated_line = f.read().strip()
        
        # Clean up temporary files
        os.unlink(temp_src_path)
        os.unlink(temp_out_path)
        
        # Detokenize the translation
        translated_tokens = translated_line.split()
        translated_text = sp_target.decode(translated_tokens)
        
        return translated_text
    
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        if 'result' in locals():
            print(f"Command output: {result.stdout}")
            print(f"Command error: {result.stderr}")
        
        # As a fallback, return the original text
        return f"[Translation error: {str(e)}] {text}"

def audio_callback(in_data, frame_count, time_info, status):
    """Callback function for PyAudio stream"""
    if stream_active:
        audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def generate_test_audio():
    """Generate synthetic audio for testing when no microphone is available"""
    # Generate 3 seconds of silent audio with occasional beep
    RATE = 16000
    duration = 3  # seconds
    samples = duration * RATE
    
    # Generate mostly silence with occasional beep
    audio_data = np.zeros(samples, dtype=np.float32)
    
    # Add a beep every second (sine wave)
    for i in range(duration):
        beep_start = i * RATE
        beep_duration = 0.1 * RATE  # 100ms beep
        for j in range(int(beep_duration)):
            t = j / RATE
            audio_data[beep_start + j] = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Convert to bytes (simulate what PyAudio would provide)
    audio_data = (audio_data * 32767).astype(np.int16).tobytes()
    return audio_data

def process_audio_stream():
    """Process audio from the queue and perform transcription/translation"""
    global asr_processor, asr_model, sp_source, sp_target
    
    RATE = 16000
    buffer = []
    last_transcript = ""
    last_transcript_time = datetime.now()
    
    while stream_active:
        try:
            # Get audio data from queue with timeout
            if app.config.get('USE_TEST_AUDIO', False):
                # Simulated audio for testing without a microphone
                data = generate_test_audio()
                # Wait a bit to simulate real-time
                threading.Event().wait(3)
            else:
                # Real microphone data
                data = audio_queue.get(timeout=0.5)
            
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.extend(audio_np)
            
            # Process when enough audio is collected
            current_time = datetime.now()
            if len(buffer) >= RATE * buffer_duration and (current_time - last_transcript_time).total_seconds() >= 1.5:
                # Process the most recent buffer_duration seconds of audio
                input_audio = np.array(buffer[-RATE * buffer_duration:])
                
                # Transcribe the audio
                transcription = transcribe_audio(asr_processor, asr_model, input_audio)
                
                if transcription and transcription.strip() and transcription != last_transcript:
                    # Translate the transcription
                    translation = translate_text(sp_source, sp_target, transcription)
                    
                    # Send the result to connected clients
                    result = {
                        'transcription': transcription.strip(),
                        'translation': translation.strip(),
                        'timestamp': current_time.strftime("%H:%M:%S")
                    }
                    
                    # Update app state with latest result
                    app.latest_result = result
                    
                    # Print for debugging
                    print(f"[{result['timestamp']}] Transcription: {result['transcription']}")
                    print(f"[{result['timestamp']}] Translation: {result['translation']}")
                    
                    last_transcript = transcription
                    last_transcript_time = current_time
                
                # Keep only last 2 seconds of audio for context
                buffer = buffer[-RATE * 2:]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio processing: {e}")
            import traceback
            print(traceback.format_exc())
            continue

# Initialize the app with latest result storage
app.latest_result = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_audio_devices')
def get_audio_devices():
    """API endpoint to get available audio devices"""
    devices = list_audio_devices()
    return jsonify({'devices': devices})

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start real-time audio streaming"""
    global stream_active, p, stream, asr_processor, asr_model, sp_source, sp_target, input_device_index
    
    try:
        if stream_active:
            return jsonify({'status': 'Stream already active'})
        
        # Get selected device index from request
        data = request.json or {}
        selected_device = data.get('deviceIndex')
        use_test_audio = data.get('useTestAudio', False)
        
        if selected_device is not None:
            input_device_index = int(selected_device)
            print(f"Using selected input device index: {input_device_index}")
        
        # Store test audio mode in app config
        app.config['USE_TEST_AUDIO'] = use_test_audio
        
        # Load models if not already loaded
        if asr_processor is None or asr_model is None:
            asr_processor, asr_model = load_asr_model()
        if sp_source is None or sp_target is None:
            _, sp_source, sp_target = load_translation_model()
        
        # Initialize PyAudio and stream if not using test audio
        if not use_test_audio:
            p = pyaudio.PyAudio()
            
            try:
                stream_kwargs = {
                    'format': pyaudio.paInt16,
                    'channels': 1,
                    'rate': 16000,
                    'input': True,
                    'frames_per_buffer': chunk_size,
                    'stream_callback': audio_callback
                }
                
                # Add device index if specified
                if input_device_index is not None:
                    stream_kwargs['input_device_index'] = input_device_index
                
                stream = p.open(**stream_kwargs)
                stream.start_stream()
            except OSError as e:
                p.terminate()
                p = None
                # If we can't initialize the audio stream, return detailed error
                print(f"PyAudio stream error: {e}")
                available_devices = list_audio_devices()
                return jsonify({
                    'error': f"Audio device error: {str(e)}",
                    'availableDevices': available_devices,
                    'suggestTestMode': True
                }), 500
        
        stream_active = True
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_audio_stream)
        processing_thread.daemon = True
        processing_thread.start()
        
        mode = "Test mode (no microphone)" if use_test_audio else "Microphone mode"
        return jsonify({'status': f'Stream started successfully - {mode}'})
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in start_stream: {error_details}")
        
        # Get list of available devices to help user troubleshoot
        available_devices = list_audio_devices()
        
        return jsonify({
            'error': str(e), 
            'details': error_details,
            'availableDevices': available_devices,
            'suggestTestMode': True
        }), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop real-time audio streaming"""
    global stream_active, p, stream
    
    try:
        if not stream_active:
            return jsonify({'error': 'No active stream'}), 400
        
        stream_active = False
        
        # Only stop actual stream if not in test mode
        if not app.config.get('USE_TEST_AUDIO', False):
            if stream:
                stream.stop_stream()
                stream.close()
            
            if p:
                p.terminate()
        
        return jsonify({'status': 'Stream stopped successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_latest_result')
def get_latest_result():
    """Get the latest transcription and translation result"""
    if app.latest_result:
        return jsonify(app.latest_result)
    else:
        return jsonify({'transcription': '', 'translation': '', 'timestamp': ''})

if __name__ == '__main__':
    # Print system info for debugging WSL/audio issues
    print(f"Python version: {sys.version}")
    print(f"Running on: {sys.platform}")
    
    # List available audio devices
    print("Available audio input devices:")
    available_devices = list_audio_devices()
    if not available_devices:
        print("WARNING: No audio input devices detected!")
        print("The application will allow testing without a microphone.")
    
    # Load models on startup
    print("Loading models on startup...")
    asr_processor, asr_model = load_asr_model()
    _, sp_source, sp_target = load_translation_model()
    print("Models loaded successfully!")
    
    # Bind to all interfaces so it's accessible from Windows host
    app.run(debug=True, host='0.0.0.0')