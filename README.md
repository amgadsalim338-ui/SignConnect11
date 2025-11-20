# SignConnect

End-to-end prototype for matching spoken audio to sign-language video labels using a shared audio–text embedding space.

## Project Structure
- `model.py`: Contrastive model definitions and save/load utilities.
- `train.py`: Training loop that builds a synthetic TTS dataset from video filenames and fits projection heads.
- `build_index.py`: Embeds text labels and saves a FAISS index plus label list.
- `app.py`: Flask application for uploading audio, embedding it, searching the FAISS index, and playing the best-match video.
- `templates/index.html`: Styled upload page and video player.
- `videos/`: Place your `.mp4` sign-language clips here (filenames determine labels).
- `outputs/`: Model and index artifacts created by training scripts.

## Setup
1. Install Python dependencies (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have system speech support for `pyttsx3` (e.g., `espeak` on Linux is typically preinstalled).

## Training Workflow
1. Add your sign-language clips under `videos/` with descriptive filenames such as `hello_how_are_you.mp4`.
2. Run training to fit the projection heads (base encoders stay frozen):
   ```bash
   python train.py --video_dir videos --output_dir outputs/model --epochs 3 --batch_size 2
   ```
   This script:
   - Discovers labels from filenames.
   - Generates TTS audio for each label (stored under `outputs/model/tts_audio`).
   - Trains the audio/text projection heads with a contrastive InfoNCE loss.
   - Saves model weights and processor/tokenizer configs to `outputs/model`.

3. Build the FAISS index for the text labels:
   ```bash
   python build_index.py --model_dir outputs/model --video_dir videos --output_dir outputs/index
   ```

## Running the Web App
After training and index creation, start the Flask server:
```bash
python app.py
```
Open `http://localhost:5000` and upload an audio clip. The app embeds the clip, searches the FAISS index, and renders the matched `.mp4` video.

## Notes
- Training is CPU-friendly because only the small projection layers are updated.
- Add or update videos any time; rerun `build_index.py` to refresh the text-label index. Retraining is recommended when you add many new phrases.
