import os
import string
import pandas as pd
import argparse
from PIL import Image
import cv2
import pytesseract
from tqdm import tqdm

from pdf2image import convert_from_path

from src.config import DPI, JPEGOPT, TESSDATA_DIR_CONFIG, OUTPUT_DIR



### For simple filename generation
def simple_counter_generator(prefix="", suffix=""):
    i=0
    while True:
        i+=1
        yield 'p'
        

def extract_info_mode1(txt):
    lines = txt.split('\n')
    eng, ind, des = '', '', ''
    data = []
    for line in lines:
        if(len(line)>1):
            flag=True
            replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            text = line.translate(replace_punctuation)
            line=text.replace('\n', ' ').replace('. ',' ')
            if(line[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ' and len(line.strip().split(' ')[0])>=2):
                if(len(ind)>0):
                    data.append([eng, ind, des])
                eng, ind, des = '', '', ''
                line = line.strip().split()
                for word in line:
                    if word[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                        eng += word + ' '
                    else:
                        ind += word + ' '
            else:
                for word in line.split():
                    if word[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                        if(len(ind)>0):
                            data.append([eng, ind, des])
                            eng, ind, des = '', '', ''
                        if flag:
                            eng, ind, des = '', '', ''
                        eng += word + ' '
                        flag=False
                    elif flag:
                        des += word + " "
                    else:
                        ind += word + ' '
    
    if(len(ind)>0):
        data.append([eng, ind, des])
        
    return data

def extract_info_mode3(txt):
    lines=txt.split('\n')
    data=[]
    sanskrit_word,english_vocab,meaning='','',''
    prev_sans_word=0
    for word in lines:
        word=word.split()
        try:
            if word[11]!='-1' and word[11].strip()[0] not in '-|:;"\'`~!@#$%^&*()-_+?<>[{()}]=1234567890':
                if int(word[6])<500:
                    if abs(int(word[7])-prev_sans_word)<10:
                        sanskrit_word+=word[11]+' '
                    else:
                        if len(sanskrit_word)>0 and len(english_vocab)>0:
                            data.append([sanskrit_word,english_vocab,meaning])
                        sanskrit_word,english_vocab,meaning=word[11],'',''
                    prev_sans_word=int(word[7])
                elif int(word[6])<900:
                    english_vocab+=word[11]+' '
                else:
                    meaning+=word[11]+' '
        except:
            pass
    if len(sanskrit_word)>0 and len(english_vocab)>0:
        data.append([sanskrit_word,english_vocab,meaning])

    return data

def extract_info_mode4(txt):
    lines=txt.split('\n')
    data=[]
    eng,ind='',''
    prev_word=0
    for line in lines:
        word=line.split()
        try:
            # if word[11]!='-1' and word[11].strip()[0] not in '-|:;"\'`~!@#$%^&*()-_+?<>[{()}]=1234567890':
            #     # print(word[11])
            #     if int(word[6])<500:
            #         if abs(int(word[7])-prev_word)<10:
            #             eng+=word[11]+' '
            #         else:
            #             if len(eng)>0 and len(ind)>0:
            #                 data.append([eng,ind])
            #             eng,ind=word[11],''
            #         prev_word=int(word[7])
            #     else:
            #         ind+=word[11]+' '
            if word[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                if len(eng)>0 and len(ind)>0:
                    data.append([eng,ind])
                    eng,ind='',''
                eng+=word+' '
            elif word[0] in '-':
                continue
            else:
                ind+=word+' '
        except:
            pass
    if len(eng)>0 and len(ind)>0:
        data.append([eng,ind])

    return data

def extract_info_mode5(txt):
    tesseract_output=txt.split('\n')
    data=[]
    eng,word1,word2,word3,word4='','','','',''
    for i in tesseract_output:
        if len(i)>1 and not i[0].isdigit():
            replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            text = i.translate(replace_punctuation)
            words=text.replace('\n', ' ').replace('. ',' ').split()
            if ord(words[0][0])<150 and len(words[0])>=2:
                if (len(eng)>0 and (len(word1)>0 or len(word2)>0) and (len(word3)>0 or len(word4)>0)):
                    data.append([eng.strip(),word1.strip(),word2.strip(),word3.strip(),word4.strip()])
                eng,word1,word2,word3,word4='','','','',''
                for word in words:
                    if word[0] in '-:"=<>{}[](*&^%\$#@!+=-_`~)\'':
                        continue
                    if ord(word[0])<150:
                        eng+=word+' '
                    elif ord(word[0])>2300 and ord(word[0])<2500:
                        word2+=word+' '
                    else:
                        word1+=word+' '
            else:
                for j in words:
                    if j[0] in '-:"=<>{}[](*&^%\$#@!+=-_`~)\'1234567890':
                        continue
                    if ord(j[0])>2300 and ord(j[0])<2500:
                        word3+=j+' '
                    else:
                        word4+=j+' '
    if (len(eng)>0 and (len(word1)>0 or len(word2)>0) and (len(word3)>0 or len(word4)>0)):
        data.append([eng.strip(),word1.strip(),word2.strip(),word3.strip(),word4.strip()])
    return data

def mode6_data_extraction(lines):
  data=[]
  col1=''
  col2=''
  i_word=''
  i_time=0
  c_word=''
  for line in lines:
    if len(line)>0:
      if i_time>1:
        i_word=''
        i_time=0
      line=line.split()
      if line[0][:3]=='---' or line[0]==',':
        if len(col1)>0 and len(col2)>0:
          data.append([col1,col2])
        col1=''
        col2=''
        flag=True
        for j in range(0,len(line)):
          try:
            if line[j][3] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ' and j==0:
              col1+=c_word+line[j][3:]+' '
              flag=False
          except:
            pass
            if line[j][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ' and j==1 and flag:
                col1+=c_word+line[j]+' '
            elif line[j][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                col1+=line[j]+' '
            elif line[j][0] not in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ-':
                col2+=line[j]+' '
          
        continue
      if line[0][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
        if len(col1)>0 and len(col2)>0:
            data.append([col1,col2])
        col1=''
        col2=''
        for index,j in enumerate(line):
          if j[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
            if j[-1]!='-' and index==0:
              col1=j+' '
              i_word=j+' '
              i_time=0
            elif j[-1]!='-':
              i_word+=j
              col1+=j+' '
            else:
              if len(i_word)>0 and len(col1)<1 and index==0:
                col1+=i_word+' '
              c_word=j[:-1]+' '
              if len(i_word)>0:
                c_word=i_word
              i_word=''
              i_time=0
              col1+=j+' '
          else:
            col2+=j+' '
      else:
        for j in line:
          col2+=j+' '
      i_time+=1

  if len(col1)>0 and len(col2)>0:
    data.append([col1,col2])
  return data

def extract_info_mode6(img_path,language,config):
    image = cv2.imread(img_path)
    img_y,img_x=image.shape[:2]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h>img_y/3 and x>img_x/3 and x<img_x*2/3:
            xl,yl,xh,yh=x,y,w,h

    # Getting the Left and Right colunns...
    h,w,_=image.shape
    left_block=[0,yl,xl,yl+yh]
    right_block=[xl+xh,yl,w,yl+yh]

    # Extracting Data ...
    l_img=image[left_block[1]:left_block[3],left_block[0]:left_block[2]]
    words=pytesseract.image_to_string(l_img,lang=language,config=config)
    word0=words.split('\n')
    output=mode6_data_extraction(word0)


    r_img=image[right_block[1]:right_block[3],right_block[0]:right_block[2]]
    words=pytesseract.image_to_string(r_img,lang=language,config=config)
    word1=words.split('\n')
    output.extend(mode6_data_extraction(word1))

    return output

def extract_info_mode7(tesseract_output,x):
  result=[]
  col1=''
  col2=''
  col3=''
  prev_word=0
  for word in tesseract_output[1:]:
      word=word.split()
      try:
          if word[11]!='-1' and word[11].strip()[0] not in ' |:;"\'`~!@#$%^&*()-_+?=1234567890.':
              if int(word[6])<x/4 and (word[11][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ,\'"/' or word[11][1] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'"/'):
                  if len(col1)>0 and len(col3)>0:
                        result.append([col1,col2,col3])
                        col1,col2,col3=word[11]+' ','',''
                  else:
                        if abs(int(word[7])-prev_word)<10:
                            col1+=word[11]+' '
                        else:
                            col1=word[11]+' '
                            col2=''
                  prev_word=int(word[7])
              else:
                    if word[11][0] in ' abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ.,\'"/' or word[11][1] in ' abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ.,\'"/':
                        col2+=word[11]+' '
                    else:
                        col3+=word[11]+' '
      except:
            pass
  if len(col1)>0 and len(col3)>0:
      result.append([col1,col2,col3])
  return result

def extract_info_mode8(txt):
    lines = txt.split('\n')
    col1, col2, col3 = '', '', ''
    data = []
    for line in lines:
        temp = ''
        replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        text = line.translate(replace_punctuation)
        line=text.replace('\n', ' ').replace('. ',' ').split()
        if(len(line)>0):
            for word in line:
                if(word[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'.,-'):
                    if(len(col2)>0 and len(col3)>0):
                        data.append([col1, col2, col3])
                        col1, col2, col3 = temp, '', ''
                        temp = ''
                    if len(col1)==0:
                        col1=temp
                        temp=''
                    col2+=word + ' '
                else:
                    temp += word +' '
            if len(col1)>0:
                col3 += temp
            else: 
                col1=temp
    
    if(len(col2)>0 and len(col1)>0):
        data.append([col1, col2, col3])
        
    return data

def mode9_data_extraction(img,language,config):
    txt = pytesseract.image_to_data(img,lang=language,config=config)
    words=txt.split('\n')
    prev_x=100000
    temp1=''
    temp2=''
    result=[]
    col1=''
    col2=''
    try:
        for i,word in enumerate(words):
            if i==0:
                continue
            word=word.split('\t')
            if len(word)==12:
                if len(word[11])>0:
                    curr_x=int(word[6])
                    if int(word[5])==1:
                        if len(temp1)>0 and len(temp2)>0 and curr_x+5<=prev_x:
                            col1+=temp1
                            col2+=temp2
                            temp1=''
                            temp2=''
                        elif len(temp1)>0 and len(temp2)>0:
                            if len(col1)>0:
                                result.append([col1,col2])
                            col1=temp1
                            col2=temp2
                            temp1=''
                            temp2=''
                        elif len(temp1)>0:
                            col1+=temp1
                            temp1=''
                        elif len(temp2)>0:
                            col2+=temp2
                            temp2=''

                        if word[11][0] in ' abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                            temp1+=word[11]+' '
                        else:
                            temp2+=word[11]+' '

                        prev_x=curr_x
                    else:
                        if word[11][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                            temp1+=word[11]+' '
                        else:
                            temp2+=word[11]+' '

        if len(temp1)>0 and len(temp2)>0:
            if len(col1)>0:
                result.append([col1,col2])
            col1=temp1
            col2=temp2
            temp1=''
            temp2=''
        elif len(temp1)>0:
            col1+=temp1
            temp1=''
        elif len(temp2)>0:
            col2+=temp2
            temp2=''
        if len(col1) and len(col2)>0:
            result.append([col1,col2])
    except:
        pass

    return result


def extract_info_mode9(img_path,language,config):
    image = cv2.imread(img_path)
    output=[]
    try:
        img_y,img_x=image.shape[:2]
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        # Detect horizontal line
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        xp,yp,xq,yq=0,0,0,0
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w>img_x*2/3 and w<=img_x and y<2*img_y/3:
                xp,yp,xq,yq=min(xp,x),max(yp,y),w,h

        # Upper Image
        img1=gray[0:yp,0:img_x]
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        detect_vertical = cv2.morphologyEx(thresh[0:yp,0:img_x], cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h>img_y/3 and x>img_x/3 and x<img_x*2/3:
                xl,yl,xh,yh=x,y,w,h
        # Getting the Left and Right colunns...
        left_block=[0,yl,xl,yl+yh]
        right_block=[xl+xh,yl,img_x,yl+yh]
        l_img=img1[left_block[1]:left_block[3],left_block[0]:left_block[2]]
        # left image
        output=mode9_data_extraction(l_img,language,config)
        r_img=img1[right_block[1]:right_block[3],right_block[0]:right_block[2]]
        # right image
        output.extend(mode9_data_extraction(r_img,language,config))

        # lower image
        img2=gray[yp:img_y,0:img_x]
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        detect_vertical = cv2.morphologyEx(thresh[yp:img_y,0:img_x], cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h>img_y/3 and x>img_x/3 and x<img_x*2/3:
                xl,yl,xh,yh=x,y,w,h
        # Getting the Left and Right colunns...
        left_block=[0,yl,xl,yl+yh]
        right_block=[xl+xh,yl,img_x,yl+yh]
        l_img=img2[left_block[1]:left_block[3],left_block[0]:left_block[2]]
        # left image
        output.extend(mode9_data_extraction(l_img,language,config))
        r_img=img2[right_block[1]:right_block[3],right_block[0]:right_block[2]]
        # right image
        output.extend(mode9_data_extraction(r_img,language,config))
    except:
        pass

    return output

def mode10_data_extraction(img,language,config):
    text = pytesseract.image_to_string(img,lang=language,config=config)
    lines=text.split('\n')
    col1=''
    temp1=''
    col2=''
    temp2=''
    result=[]
    for line in lines:
        replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        text = line.translate(replace_punctuation)
        line=text.replace('\n', ' ').replace('. ',' ').split()
        if len(line)>0:
            if line[0][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                for word in line:
                    if word[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
                        temp1+=word+' '
                    else:
                        temp2+=word+' '
            else:
                temp2=' '.join(line)

            if len(temp1)>0 and len(temp2)>0:
                if len(col1)>0:
                    result.append([col1,col2])
                col1=temp1
                col2=temp2
                temp1=''
                temp2=''
            elif len(temp1)>0:
                col1+=temp1
                temp1=''
            elif len(temp2)>0:
                col2+=temp2
                temp2=''

    if len(temp1)>0 and len(temp2)>0:
        if len(col1)>0:
            result.append([col1,col2])
        col1=temp1
        col2=temp2
        temp1=''
        temp2=''
    elif len(temp1)>0:
        col1+=temp1
        temp1=''
    elif len(temp2)>0:
        col2+=temp2
        temp2=''
    if len(col1) and len(col2)>0:
        result.append([col1,col2])

    return result


def extract_info_mode10(img_path,language,config):
    img=cv2.imread(img_path)
    output=[]
    try:
        img_y,img_x=img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h>img_y/3 and x>img_x/3 and x<img_x*2/3:
                xl,yl,xh,yh=x,y,w,h
        # Getting the Left and Right colunns...
        left_block=[0,yl,xl,yl+yh]
        right_block=[xl+xh,yl,img_x,yl+yh]
        # left image
        l_img=img[left_block[1]:left_block[3],left_block[0]:left_block[2]]
        output=mode10_data_extraction(l_img,language,config)
        # right image
        r_img=img[right_block[1]:right_block[3],right_block[0]:right_block[2]]
        output.extend(mode10_data_extraction(r_img,language,config))
    except:
        pass

    return output

def extract_info_mode11(txt):
    lines = txt.split('\n')
    tamil_range = (0x0B80, 0x0BFF)
    col1, col2, col3 = '', '', ''
    data = []
    for line in lines:
        temp = ''
        replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        text = line.translate(replace_punctuation)
        line=text.replace('\n', ' ').replace('. ',' ').split()
        if(len(line)>0):
            for word in line:
                # print('FIRST WORD :',word[0])
                if(word[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'.,-'):
                    if(len(col1)>0 and len(col2)>0 and len(col3)>0):
                        # print([col1,col2,col3])
                        data.append([col1, col2, col3])
                        col1, col2, col3 = '', '', ''
                    if len(col1)==0:
                        col1=word+' '
                    else:
                        col1+=word + ' '
                elif tamil_range[0] <= ord(word[0]) <= tamil_range[1]:
                    if len(col3)==0:
                        col3=word+' '
                    else:
                        col3+=word + ' '
                else:
                    if len(col2)==0:
                        col2=word+' '
                    else:
                        col2+=word+' '
    if(len(col1)>0 and len(col2)>0 and len(col3)>0):
        data.append([col1, col2, col3]) 

    return data

def extract_mode12_data(tesseract_output, x):
    line = -1
    para = -1
    col1=''
    temp1=''
    col2=''
    temp2=''
    col3=''
    temp3=''
    result=[]
    for word in tesseract_output.split('\n')[1:]:
        word=word.split()
        try:
            curr_line = int(word[4])
            curr_para = int(word[3])
            if word[11]!='-1' and word[11].strip()[0] not in ' |:;"\'`~!@#$%^&*()-_+?=1234567890.':
                if line!=curr_line or curr_para!=para:
                    line = curr_line
                    para = curr_para
                    if len(temp1)>0 and len(temp3)>0:
                        if len(col1)>0 and len(col3)>0:
                            result.append([col1,col2,col3])
                            # print([col1,col2,col3], 'added')
                        col1=temp1
                        col2=temp2
                        col3=temp3
                        temp1 = ''
                        temp2 = ''
                        temp3 = ''
                    elif len(temp1)>0:
                        col1+=temp1
                        temp1 = ''
                    elif len(temp2)>0:
                        col2+=temp2
                        temp2 = ''
                    elif len(temp3)>0:
                        col3+=temp3
                        temp3 = ''
                # print('word :', word[11])
                if (word[11][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ,\'"/' or word[11][1] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'"/'):
                    temp1 += word[11] + ' '
                elif int(word[6])<5*x/8 and (word[11][0] not in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ,\'"/' or word[11][1] not in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'"/'):
                    temp2 += word[11] + ' '
                elif (word[11][0] not in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ,\'"/' or word[11][1] not in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'"/'):
                    temp3 += word[11] + ' '
                # print('temp :', [temp1, temp2, temp3])
                # print('col :', [col1,col2,col3])


        except Exception as e:
            # print(e)
            # print(word)
            pass
    
    if len(col1)>0 and len(col3)>0:
        result.append([col1,col2,col3])
        # print([col1,col2,col3], 'added')

    if len(temp1)>0 and len(temp3)>0:
        result.append([temp1, temp2, temp3])
        # print([temp1,temp2,temp3], 'added')
    return result

def extract_info_mode12(img_path,language,config):
    img=cv2.imread(img_path)
    y,x=img.shape[:2]
    tesseract_output=pytesseract.image_to_data(img,lang=language,config=config)
    result=extract_mode12_data(tesseract_output,x)
    return result

def extract_mode13_data(tesseract_output, x):
    line = -1
    para = -1
    col1=''
    temp1=''
    col3=''
    temp3=''
    result=[]
    for word in tesseract_output.split('\n')[1:]:
        word=word.split()
        try:
            curr_line = int(word[4])
            curr_para = int(word[3])
            if word[11]!='-1' and word[11].strip()[0] not in ' |:;"\'`~!@#$%^&*()-_+?=1234567890.':
                if line!=curr_line or curr_para!=para:
                    line = curr_line
                    para = curr_para
                    if len(temp1)>0 and len(temp3)>0:
                        if len(col1)>0 and len(col3)>0:
                            result.append([col1,col3])
                            # print([col1,col3], 'added')
                        col1=temp1
                        col3=temp3
                        temp1 = ''
                        temp3 = ''
                    elif len(temp1)>0:
                        col1+=temp1
                        temp1 = ''
                    elif len(temp3)>0:
                        col3+=temp3
                        temp3 = ''
                # print('word :', word[11])
                if (word[11][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ,\'"/' or word[11][1] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ\'"/'):
                    temp1 += word[11] + ' '
                else:
                    temp3 += word[11] + ' '
                # print('temp :', [temp1, temp3])
                # print('col :', [col1,col3])


        except Exception as e:
            print(e)
            # print(word)
            pass
    if len(col1)>0 and len(col3)>0:
        result.append([col1,col3])
        # print([col1,col2,col3], 'added')

    if len(temp1)>0 and len(temp3)>0:
        result.append([temp1, temp3])
        # print([temp1,temp2,temp3], 'added')
    return result

def extract_info_mode13(img_path,language,config):
    img=cv2.imread(img_path)
    y,x=img.shape[:2]
    tesseract_output=pytesseract.image_to_data(img,lang=language,config=config)
    result=extract_mode13_data(tesseract_output,x)
    return result

def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    
    images_folder = os.path.join(OUTPUT_DIR, args.images_folder_name, 'images')
    out_file = os.path.join(OUTPUT_DIR, args.images_folder_name, 'results.csv')

    print('Starting')
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        output_file=simple_counter_generator("page",".jpg")
        convert_from_path(args.orig_pdf_path ,output_folder=images_folder, dpi=DPI,fmt='jpeg',jpegopt=JPEGOPT,output_file=output_file)
 
 
    # images_folder = './outputs/test/images/'
    img_files = sorted(os.listdir(images_folder))
    
    
    result = []
    
    curr_page = 1
    for img_file in tqdm(img_files):
        
        if(args.start_page is not None and curr_page < int(args.start_page)):
            curr_page += 1
            continue
        
        elif(args.end_page is not None and curr_page > int(args.end_page)):
            break
        
        else:
            curr_page += 1
            
        img_path = os.path.join(images_folder, img_file)
        image = Image.open(img_path)
        gray_image = image.convert('L')
        x,y=gray_image.size
        
        if args.mode=='1':
            txt = pytesseract.image_to_string(gray_image, lang=args.language_model, config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode1(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='2':
            txt = pytesseract.image_to_string(gray_image, lang=args.language_model, config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode1(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='3':
            txt = pytesseract.image_to_string(gray_image,lang=args.language_model,config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode3(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='4':
            txt = pytesseract.image_to_data(gray_image,lang=args.language_model,config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode4(txt)
                result.extend(value)
            except:
                pass
        
        elif args.mode=='5':
            txt = pytesseract.image_to_string(gray_image,lang=args.language_model,config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode5(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='6':
            try:
                value=extract_info_mode6(img_path,args.language_model,TESSDATA_DIR_CONFIG)
                result.extend(value)
            except:
                pass

        elif args.mode=='7':
            txt = pytesseract.image_to_data(gray_image,lang=args.language_model,config=TESSDATA_DIR_CONFIG)
            try:
                value=extract_info_mode7(txt,x)
                result.extend(value)
            except:
                pass
        
        elif args.mode=='8':
            txt = pytesseract.image_to_string(gray_image, lang=args.language_model, config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode8(txt)
                result.extend(value)
            except:
                pass
        
        elif args.mode=='9':
            try:
                value=extract_info_mode9(img_path,args.language_model,TESSDATA_DIR_CONFIG)
                result.extend(value)
            except:
                pass
        
        elif args.mode=='10':
            try:
                value=extract_info_mode10(img_path,args.language_model,TESSDATA_DIR_CONFIG)
                result.extend(value)
            except:
                pass
        
        elif args.mode=='11':
            txt = pytesseract.image_to_string(gray_image, lang=args.language_model, config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode11(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='12':
            try:
                value=extract_info_mode12(img_path,args.language_model,TESSDATA_DIR_CONFIG)
                print(value)
                result.extend(value)
            except:
                pass

        elif args.mode=='13':
            try:
                value=extract_info_mode13(img_path,args.language_model,TESSDATA_DIR_CONFIG)
                result.extend(value)
            except:
                pass
        
    if args.mode == '1':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning','Indic Description'])
    elif args.mode == '2':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning','Indic Description'])
    elif args.mode == '3':
        df = pd.DataFrame(result, columns = ['Sanskrit Word','English Vocab','Description'])
    elif args.mode == '4':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning'])
    elif args.mode == '5':
        df = pd.DataFrame(result, columns = ['English Word','Word 1','Word 2','Word 3','Word 4'])
    elif args.mode == '6':
        df = pd.DataFrame(result, columns = ['English Word','Oriya Word'])
    elif args.mode == '7':
        df = pd.DataFrame(result, columns = ['Col1','Col2','Col3'])
    elif args.mode == '8':
        df = pd.DataFrame(result, columns = ['Col1','Col2','Col3'])
    elif args.mode == '9':
        df = pd.DataFrame(result, columns = ['col1','col2'])
    elif args.mode == '10':
        df = pd.DataFrame(result, columns = ['col1','col2'])
    elif args.mode == '11':
        df = pd.DataFrame(result, columns = ['col1','col2','col3'])
    elif args.mode == '12':
        df = pd.DataFrame(result, columns = ['col1','col2','col3'])
    elif args.mode == '13':
        df = pd.DataFrame(result, columns = ['col1','col2'])
    
    
    print('saving the result at',out_file)
    df.to_csv(out_file, index=False)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--orig_pdf_path", type=str, default=None, help="path to the input pdf file")
    parser.add_argument("-im", "--images_folder_name", type=str, default="pdf", help="type of input file | pdf/images")
    parser.add_argument("-l", "--language_model", type=str, default="eng+hin", help="language to be used for OCR")
    parser.add_argument("-m", "--mode", type=str, default='1', help="mode 1 => 1 , mode 2 => 2 , Eng-Sanskrit => 3 , bhandaran => 4, English_Hindi_Tamil => 5 , English_Oriya => 6 , English_English_Hindi => 7 , Hindi_English_Hindi => 8")
    parser.add_argument("-s", "--start-page", type=str, default=None, help="Start page for OCR")
    parser.add_argument("-e", "--end-page", type=str, default=None, help="End page for OCR")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
