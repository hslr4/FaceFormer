import uvicorn
import argparse
import os
import pickle
from faceformer import Faceformer
import numpy as np
import librosa
from transformers import Wav2Vec2Processor
from faceformer import PeriodicPositionalEncoding, init_biased_mask

from fastapi import FastAPI, Response, Request

import torch


WAV_PATH = 'testapi.wav'

parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
parser.add_argument("--model_name", type=str, default="vocaset")
parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
parser.add_argument("--condition", type=str, default="FaceTalk_170913_03279_TA", help='select a conditioning subject from train_subjects')
parser.add_argument("--subject", type=str, default="FaceTalk_170809_00138_TA", help='select a subject from test_subjects or train_subjects')
parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
parser.add_argument("--max_seq_len", type=int, default=6000, help='maximum input sequence length')#default=600

parser.add_argument("--host", type=str, default='0.0.0.0', help='host to expose rest api')
parser.add_argument("--port", type=int, default=7222, help='port to expose rest api')
#parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
args = parser.parse_args()   



with torch.no_grad():
    #build model
    model = Faceformer(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name))))
    model.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period, max_seq_len=args.max_seq_len)
    model.biased_mask = init_biased_mask(n_head = 4, max_seq_len=args.max_seq_len, period=args.period)
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]
                
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Lipsync"}


@app.post("/lipsync/", response_class=Response)
async def lipsync(data: Request):
    data_b = await data.body()
    with open(WAV_PATH, mode='bw') as f:
        f.write(data_b)

    with torch.no_grad():
        speech_array, _ = librosa.load(os.path.join(WAV_PATH), sr=16000)
        audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature)
        err = False
        try:
            prediction = model.predict(audio_feature.to(device=args.device), template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            prediction = prediction.detach().cpu().numpy()
        except Exception as e:
            print(e)
            prediction = None
            err = True
        torch.cuda.empty_cache()
        if err:
            raise Exception('failed to predict vertices, see previous print')
        return Response(content=prediction.tobytes(), media_type='application/octet-stream')


if __name__ == '__main__':
    uvicorn.run(app, port=args.port, host=args.host)