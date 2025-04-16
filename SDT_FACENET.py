from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# - set margin=20 to include more background around the face, helping FaceNet handle a very tight crop
# - lowered thresholds to [0.2, 0.3, 0.4] to make detection stages more lenient, increasing the chance of detecting obscured faces
# changes were chosen based on visual inspection of failed detections in images
mtcnn = MTCNN(image_size=160, margin=20, thresholds=[0.2, 0.3, 0.4], post_process=True)

FILE_PATH = '/Users/codylejang/Desktop/eyewitness/'

# # example image to output
# image_path = '/Users/codylejang/Desktop/eyewitness/innocent13.png'
# output_path = '/Users/codylejang/Desktop/face_crop_example.png'

# # load image
# img = Image.open(image_path).convert('RGB')

# # MTCNN to detect and crop the face
# face_tensor = mtcnn(img)
# # if values are between -1 and 1 need to scale due to conversion to pil image
# face_tensor = (face_tensor + 1) / 2  # scale to [0,1]
# face_tensor = torch.clamp(face_tensor, 0, 1) # clip if necessary
# to_pil = transforms.ToPILImage()
# face_pil = to_pil(face_tensor.cpu())
# face_pil.save('/Users/codylejang/Desktop/face_crop_example.png')

def get_embedding(image_path):
    #image preprocessing for facenet
    img = Image.open(image_path).convert('RGB')
    face_tensor = mtcnn(img)
    if face_tensor is None:
        print(f"Face not detected in {image_path}")
        return None
    
    with torch.no_grad():
        face = face_tensor.unsqueeze(0).to(device)
    embedding = facenet(face)

    return embedding.detach().cpu().numpy()[0]

def run_trials():
    # path to master csv with all trials
    all_trials_path = os.path.join(FILE_PATH, 'eyewitness_trials.csv')
    loadouts = pd.read_csv(all_trials_path)
    
    error_log = []
    results = []
    for i, trial in loadouts.iterrows():
        # get embeddings from paths to all images
        encoding_image = get_embedding(os.path.join(FILE_PATH, trial.loc['encoding_image']))
        target_image = get_embedding(os.path.join(FILE_PATH, trial.loc['target_image']))
        innocent_image = get_embedding(os.path.join(FILE_PATH, trial.loc['innocent_image']))
        left_image = get_embedding(os.path.join(FILE_PATH, trial.loc['left_image']))
        right_image = get_embedding(os.path.join(FILE_PATH, trial.loc['right_image']))
        correct_position = trial['correct_position'].strip().lower()
        
        #logging undetected faces or errors
        if any(x is None for x in [encoding_image, target_image, innocent_image]):
            for image_name, image in zip(['encoding', 'target', 'innocent'], [encoding_image, target_image, innocent_image]):
                if image is None:
                    error_log.append(f"{i} - {image_name} face not detected")
            continue
        
        # compute distances 
        dist_left = np.linalg.norm(encoding_image - left_image)
        dist_right = np.linalg.norm(encoding_image - right_image)
        predicted = 'left' if dist_left < dist_right else 'right'
        correct = (predicted == correct_position)
        
        results.append({
            'trial': i,
            'predicted': predicted,
            'correct_position': correct_position,
            'accuracy': int(correct),
            'dist_left': dist_left,
            'dist_right': dist_right
        })
    
    return pd.DataFrame(results), error_log

results_df, errors = run_trials()
print(results_df)
print(f"\nErrors: {len(errors)}")




        




