import os
import sys
import torch
import numpy as np
from hyperpyyaml import load_hyperpyyaml
import torchaudio
import argparse

class Inference:
    def __init__(self, model_path, CUDA=True):
        with open('args.yaml') as file:
            args = load_hyperpyyaml(file)
        device_num = args['device']
        if CUDA and torch.cuda.is_available():
            torch.cuda.set_device(device_num)
            self.device = 'cuda:' + str(device_num)
        else:
            self.device = 'cpu'
        self.sr_model = args['sr']
        self.classes = args['classes']
        self.threshold = args['threshold']
        self.feat_extractor = args['feature_extractor']
        self.model = torch.nn.ModuleDict({ 'encoder': args['encoder'], 'decoder': args['decoder']}) 
        self.feat_extractor.to(self.device)
        self.model.to(self.device)
        self.load_model(model_path)
        print('Pretrained model has been restored')
        
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def forward(self, signal):
        self.model.eval()
        self.feat_extractor.eval()
        with torch.no_grad():
            features = self.feat_extractor(signal)
            encoded_features = self.model.encoder(features)
            predictions = self.model.decoder(encoded_features)
            
        return predictions
    
    def device_change(self, new_device):
        if new_device > -1 and torch.cuda.is_available():
            torch.cuda.set_device(new_device)
            self.device = 'cuda:' + str(new_device)
        else:
            self.device = 'cpu'
        self.feats_extractor.to(self.device)
        self.model.to(self.device)
    
      
    def predict(self, signal, sample_rate):
        if signal.shape[0] == 2:
            signal = signal.mean(axis=0)
        if len(signal.shape) == 1:
            signal = signal[None, :]
        signal = signal.to(self.device)
        if sample_rate != self.sr_model:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sr_model)
            resampler.to(signal.device)
            signal = resampler(signal)   

        predictions = self.forward(signal).cpu().numpy()[0]
        results = {}
        for i, class_ in enumerate(self.classes):
            results[class_] = predictions[i]
        result_pred = np.argwhere(predictions > self.threshold).tolist()
        if len(result_pred) == 0:
            result_pred_str = 'Ничего из 4х классов на аудиосигнале не обнаружено'
        else:
            result_pred_str = 'На аудиосигнале обнаружено: ' + ', '.join(np.array(self.classes)[result_pred])
        results_dict = {'predictions': predictions, 'dict_preds': results, 'result_pred_str': result_pred_str}

        return results_dict
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Введите путь до модели и путь до аудиосигнала')
    parser.add_argument('--MODEL_PATH', type=str, help='Путь до предобученной модели')
    parser.add_argument('--AUDIO_PATH', type=str, help='Путь до аудиосигнала')
    parser.add_argument('--CUDA', default=False, type=bool, help='Использовать CUDA')
    args_inference = parser.parse_args()
    inference = Inference(args_inference.MODEL_PATH, args_inference.CUDA)   
    audio, sr = torchaudio.load(args_inference.AUDIO_PATH)
    results = inference.predict(audio, sr)
    print('\nРЕЗУЛЬТАТ:\n')
    for key in results['dict_preds'].keys():
        print('{}: {:.2f}'.format(key, results['dict_preds'][key]))
    
    print('\n{}\n'.format(results['result_pred_str']))
