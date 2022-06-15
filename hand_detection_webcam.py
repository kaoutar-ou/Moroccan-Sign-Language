from hashlib import new
import cv2
import mediapipe as mp
import joblib
import numpy as np
from pyrsistent import v
from sqlalchemy import null

import emoji
from variables import *
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image

import gtts
from playsound import playsound
import io
import pyttsx3

import threading
import time

engine = pyttsx3.init()
engine.setProperty('rate', 105)
engine.setProperty('voice', 1)

import os

import wave



from pydub import AudioSegment

from moviepy.editor import concatenate_audioclips, AudioFileClip



res=''
res1=''
mp3_file = 0
output_mp3_file = 0
rep_letters = 'C:\\Users\\ismail\\Desktop\\Agile\\morr\\Moroccan-Sign-Language\\history\\letters\\voice'
rep_words = 'C:\\Users\\ismail\\Desktop\\Agile\\morr\\Moroccan-Sign-Language\\history\\words\\voice'

def speak(text,path,mp3) :
    # rep = 'C:\\Users\\kaout\\irisi4\\MA\\mp\\220522_2\\MSL\\history\\voice'
    mp3_ = io.BytesIO()
    tts = gtts.gTTS(text=text, lang='ar')
    # tts.write_to_fp(mp3_)
    name = path+str(mp3)+'.mp3'
    tts.save(name)
    playsound(name)
    # os.remove(name)


    # tts.save("C:/Users/kaout/irisi4/MA/mp/220522_2/MSL/voice.mp3")
    # playsound("C:/Users/kaout/irisi4/MA/mp/220522_2/MSL/voice.mp3")


def concatenate_audio_moviepy(audio_clip_paths, output_path):
    # clips = [AudioFileClip(c) for c in './history']

    print(clips)
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path)


add_letter = False
sentence = ''
reshaped_text = arabic_reshaper.reshape(" ")



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)

cap = cv2.VideoCapture(0)


def data_clean(landmark):

    data = landmark[0]

    try:
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        return([clean])

    except:
        return(np.zeros([1, 63], dtype=int)[0])


mphands = mp.solutions.hands
while cap.isOpened():
   
    success, image = cap.read()

    h, w, c = image.shape

    h += 50
    w += 50

    image = cv2.flip(image, 1)

    if not success:
        break

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
    # if results.multi_hand_landmarks is not None:
        # for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
        #                                         results.multi_handedness):
            # print(results.multi_handeness)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                x_min -= 5
                y_min -= 5
            cv2.rectangle(image, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
            # mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)

        cleaned_landmark = data_clean(results.multi_hand_landmarks)
        # print(cleaned_landmark)

        # LABELS = {'A':'أ','AIN':'ع','B':'ب','D':'د','DAD':'ض','GH':'غ','L':'ل','R':'ر','S':'س','T':'ت','Y':'ي'}

        if cleaned_landmark:
            clf = joblib.load('model/msl_model_4.pkl')
            y_pred = clf.predict(cleaned_landmark)

            # print(y_pred)
            
            # ppp = y_pred * 100

            # print(y_pred)

            # most_likely_class_index = int(np.argmax(y_pred))

            # y_pred *= 100

            # if y_pred[0] in LABELS.values() :

            # res1 = LABELS[y_pred[0]]
            res = LABELS[y_pred[0]]

            if res != res1 :
                # time.sleep(2).
                # print(y_pred)
                res1 = res

                mp3_file+=1
                if (res1== ';('):
                    th = threading.Thread(target=speak, args=('انا حزين', rep_letters, mp3_file))
                elif(res1==':('):
                    th = threading.Thread(target=speak, args=('انا غاضب', rep_letters, mp3_file))
                elif(res1==':)'):
                    th = threading.Thread(target=speak, args=('انا سعيد', rep_letters, mp3_file))
                else:
                    th = threading.Thread(target=speak, args=(res, rep_letters, mp3_file))
                # print(mp3_file)
                th.start()
                
            c = cv2.waitKey(1) & 0xff

            if c == ord('n') or c == ord('N'):
                sentence += res

            if c == ord('c') or c == ord('C'):
                sentence = ''
            
            if c == ord('r') or c == ord('R'):
                # print('rrr')
                if sentence != '' :
                    print(sentence)
                    mp3_file+=1
                    th = threading.Thread(target=speak, args=(sentence, rep_words, mp3_file))
                    th.start()
            
            combined_letters = []
            combined_words = []

            last_letter = ''
            last_word = ''

            index_letter = 0
            index_word = 0

            if c == ord('s') or c == ord('S'):
                for file in os.listdir('history/letters') :
                    sound = AudioFileClip("./history/letters/"+file)
                    combined_letters.append(sound)
                
                for file in os.listdir('output_mp3/letters') :
                    last_letter = file

                if last_letter != '' :
                    i_letter = last_letter.split('_')[1].split('.')[0]
                    index_letter = int(i_letter) + 1

                final_clip_letters = concatenate_audioclips(combined_letters)
                final_clip_letters.write_audiofile('./output_mp3/letters/voice_'+str(index_letter)+'.mp3')

                for file in os.listdir('history/words') :
                    sound = AudioFileClip("./history/words/"+file)
                    combined_words.append(sound)

                for file in os.listdir('output_mp3/words') :
                    last_word = file
                
                if last_word != '' :
                    i_word = last_word.split('_')[1].split('.')[0]
                    index_word = int(i_word) + 1

                final_clip_words = concatenate_audioclips(combined_words)
                final_clip_words.write_audiofile('./output_mp3/words/voice_'+str(index_word)+'.mp3')

                # output_mp3_file += 1

            if c == ord('i') or c == ord('I'):
                for file in os.listdir('history/letters') :
                    os.remove('history/letters/'+file)
                for file in os.listdir('history/words') :
                    os.remove('history/words/'+file)

            if res == 'أ':
                add_letter = True

            if add_letter :
                if res != 'أ' :
                    sentence += res
                    add_letter = False

            reshaped_text_sentence = arabic_reshaper.reshape(sentence)

            reshaped_text = arabic_reshaper.reshape(res)

            bidi_text = get_display(reshaped_text)
            bidi_text_sentence = get_display(reshaped_text_sentence)

            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath, 52)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            if(bidi_text==';('):
                logo = cv2.imread("/home/akihiki/PycharmProjects/Moroccan-Sign-Language/emojis/cry-face.png")
                logo_pil = Image.fromarray(logo)
                img_pil.paste(logo_pil, (x_max, y_min))

                draw.text((x_min, y_min), 'انا حزين', font=font)
            elif(bidi_text==':('):
                logo = cv2.imread("/home/akihiki/PycharmProjects/Moroccan-Sign-Language/emojis/angry-face.png")
                logo_pil = Image.fromarray(logo)
                img_pil.paste(logo_pil, (x_max, y_min))

                draw.text((x_min, y_min), 'انا غاضب', font=font)

            elif(bidi_text==':)'):
                logo = cv2.imread("/home/akihiki/PycharmProjects/Moroccan-Sign-Language/emojis/smilling-face.png")
                logo_pil = Image.fromarray(logo)
                img_pil.paste(logo_pil, (x_max, y_min))

                draw.text((x_min, y_min), 'انا سعيد', font=font)
            else:
                draw.text((x_min, y_min), bidi_text, font=font)
                draw.text((10,30), bidi_text_sentence, font=font)

            image = np.array(img_pil)

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        for file in os.listdir('history/letters') :
            os.remove('history/letters/'+file)
        for file in os.listdir('history/words') :
            os.remove('history/words/'+file)
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
