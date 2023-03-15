# server.py
# we will use asyncio to run our service
import asyncio
import grpc

import cv2
import numpy as np

import torch
from torchvision import transforms
from emotic import Emotic

# from the generated grpc server definition, import the required stuff
from ms_emotionRecognition_pb2_grpc import EmotionRecognitionService, add_EmotionRecognitionServiceServicer_to_server
# import the requests and reply types
from ms_emotionRecognition_pb2 import EmotionRecognitionRequest, EmotionRecognitionInferenceReply

from PIL import Image
import os

import logging
from time import perf_counter


logging.basicConfig(level=logging.INFO)

class EmotionRecognitionService(EmotionRecognitionService):

    def __init__(self) -> None:
        self.device = torch.device("cuda:%s" %(str(0)) if torch.cuda.is_available() else "cpu")
        self.model_context = torch.load(os.path.join("models/emotic",'model_context1.pth')).to(self.device)
        self.model_body = torch.load(os.path.join("models/emotic",'model_body1.pth')).to(self.device)
        #emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
        self.emotic_state_dict = torch.load(os.path.join("models/emotic",'model_emotic1.pt'))
        super().__init__()

    async def ProcessImagesForEmotic(self, context_norm, body_norm, npimg, image_context=None, image_body=None, bbox=None):
        ''' Prepare context and body image. 
        :param context_norm: List containing mean and std values for context images. 
        :param body_norm: List containing mean and std values for body images. 
        :param image_context_path: Path of the context image. 
        :param image_context: Numpy array of the context image.
        :param image_body: Numpy array of the body image. 
        :param bbox: List to specify the bounding box to generate the body image. bbox = [x1, y1, x2, y2].
        :return: Transformed image_context tensor and image_body tensor.
        '''
        
        image_context = npimg[...,::-1].copy()

        if bbox is not None:
            image_body = image_context[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
        
    
        image_context = cv2.resize(image_context, (224,224))
        image_body = cv2.resize(image_body, (128,128))
    
        
        test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
        context_norm = transforms.Normalize(context_norm[0], context_norm[1])  
        body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        image_context = context_norm(test_transform(image_context)).unsqueeze(0)
        image_body = body_norm(test_transform(image_body)).unsqueeze(0)

        return image_context, image_body 


    async def Inference(self, request: EmotionRecognitionRequest, context) -> EmotionRecognitionInferenceReply:
        start = perf_counter()

        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        
        thresholds_path = "thresholds"
    
        cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
            'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
            'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
        cat2ind = {}
        ind2cat = {}

        for idx, emotion in enumerate(cat):
            cat2ind[emotion] = idx
            ind2cat[idx] = emotion

        vad = ['Valence', 'Arousal', 'Dominance']
        ind2vad = {}
        for idx, continuous in enumerate(vad):
            ind2vad[idx] = continuous

        context_mean = [0.4690646, 0.4407227, 0.40508908]
        context_std = [0.2514227, 0.24312855, 0.24266963]
        body_mean = [0.43832874, 0.3964344, 0.3706214]
        body_std = [0.24784276, 0.23621225, 0.2323653]
        context_norm = [context_mean, context_std]
        body_norm = [body_mean, body_std]

        
        thresholds = torch.FloatTensor(np.load(os.path.join(thresholds_path, 'emotic_thresholds.npy'))).to(self.device) 
    
        '''
        model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
        model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
        #emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
        emotic_state_dict = torch.load(os.path.join(model_path,'model_emotic1.pt'))
        '''

        model_context = self.model_context
        model_body = self.model_body
        #emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
        emotic_state_dict = self.emotic_state_dict

        #https://github.com/pytorch/pytorch/issues/7812
        #https://pytorch.org/docs/master/notes/serialization.html

        emotic_model = Emotic(2048,2048)
        #emotic_model = Emotic(512,512)
        emotic_model.load_state_dict(emotic_state_dict)

        model_context.eval()
        model_body.eval()
        emotic_model.eval()

        bbox = [int(request.personBox["x1"]), int(request.personBox["y1"]), int(request.personBox["x2"]), int(request.personBox["y2"])] # x1 y1 x2 y2
        
        #im = Image.fromarray(npimg)
        #im.save(os.getcwd() + "/original.png")
        #im = Image.fromarray(npimg[request.personBox['y1']:request.personBox['y2'], request.personBox['x1']:request.personBox['x2']])
        #im.save(os.getcwd() + "/frame.png")

        image_context = None
        image_body = None
        image_context, image_body = await self.ProcessImagesForEmotic(context_norm, body_norm, npimg, image_context=image_context, image_body=image_body, bbox=bbox)
        
        with torch.no_grad():
            image_context = image_context.to(self.device)
            image_body = image_body.to(self.device)
            
            pred_context = model_context(image_context)
            pred_body = model_body(image_body)
            pred_cat, pred_cont = emotic_model(pred_context, pred_body)
            pred_cat = pred_cat.squeeze(0)
            pred_cont = pred_cont.squeeze(0).to("cpu").data.numpy()

            bool_cat_pred = torch.gt(pred_cat, thresholds)
        

        result = {'continuous' : {}, 'categorical' : []}

        for i in range(len(bool_cat_pred)):
            if bool_cat_pred[i] == True:
                result['categorical'].append(ind2cat[i])

        pred_cont = 10*pred_cont

        count = 0
        for continuous in pred_cont:
            result['continuous'][vad[count]] = continuous # Valence Arousal Dominance
            count += 1

        logging.info(
            f"[✅] In {(perf_counter() - start) * 1000:.2f}ms"
        )


        return EmotionRecognitionInferenceReply(continuous=result['continuous'], categorical=result['categorical'])


      

async def serve():
    server = grpc.aio.server()
    add_EmotionRecognitionServiceServicer_to_server(EmotionRecognitionService(), server)
    # using ip v6
    adddress = "[::]:50055"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())