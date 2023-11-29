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
        if(len(line)>4 and line[0] not in string.punctuation and not line[0].isdigit()):
            if(ord(line[0])<150 and len(line.strip().split(' ')[0])>=2):
                if(len(ind)>3):
                    data.append([eng, ind, des])
                eng, ind, des = '', '', ''
                line = line.strip().split()
                eng_word_flag=True
                for word in line:
                    if(word in '-|:;"\'`~!@#$%^&*()-_+?<>[{()}]=1234567890'):
                        continue
                    elif(ord(word[0])<120 and eng_word_flag) :
                        eng += word + ' '
                    else:
                        ind += word + ' '
                        eng_word_flag=False
            else:
                des += line
    
    if(len(ind)>3):
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
                # print(word[11])
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
    for word in lines:
        word=word.split()
        try:
            if word[11]!='-1' and word[11].strip()[0] not in '-|:;"\'`~!@#$%^&*()-_+?<>[{()}]=1234567890':
                # print(word[11])
                if int(word[6])<500:
                    if abs(int(word[7])-prev_word)<10:
                        eng+=word[11]+' '
                    else:
                        if len(eng)>0 and len(ind)>0:
                            data.append([eng,ind])
                        eng,ind=word[11],''
                    prev_word=int(word[7])
                else:
                    ind+=word[11]+' '
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
            words=i.split()
            if ord(words[0][0])<150 and len(words[0])>=2 and words[0][0] not in string.punctuation:
                if (len(eng)>1 and len(word1)>1 and len(word2)>1 and len(word3)>1):
                    data.append([eng.strip(),word1.strip(),word2.strip(),word3.strip(),word4.strip()])
                    # print("->>>",[eng,word1,word2,word3,word4])
                eng,word1,word2,word3,word4='','','','',''
                for j in words:
                    # print(j,ord(j[0]))
                    if j[0] in '-:"=<>{}[](*&^%\$#@!+=-_`~)\'1234567890':
                        continue
                    if ord(j[0])<150:
                        eng+=j+' '
                    elif ord(j[0])>2300 and ord(j[0])<2900:
                        for h in j:
                            if h not in '\u200c':
                                word2+=h
                        word2+=' '
                    else:
                        for h in j:
                            if h not in '\u200c':
                                word1+=h
                        word1+=' '
            else:
                for j in words:
                    # print(j,ord(j[0]))
                    if j[0] in '-:"=<>{}[](*&^%\$#@!+=-_`~)\'1234567890':
                        continue
                    if ord(j[0])>2300 and ord(j[0])<2900:
                        for h in j:
                            if h not in '\u200c':
                                word3+=h
                        word3+=' '
                    else:
                        for h in j:
                            if h not in '\u200c':
                                word4+=h
                        word4+=' '
    if (len(eng)>1 and len(word1)>1 and len(word2)>1 and len(word3)>1):
        data.append([eng.strip(),word1.strip(),word2.strip(),word3.strip(),word4.strip()])
        # print("->>>",[eng,word1,word2,word3,word4])
    return data

def mode6_data_extraction(word):
  data=[]
  eng_word=''
  ori_word=''
  i_word=''
  i_time=0
  c_word=''
  for i in word:
    if len(i)>1:
      if i_time>1:
        i_word=''
        i_time=0
      i=i.split()
      if i[0][:3]=='---' or i[0]==',':
        # print('data',[eng_word,ori_word])
        if len(eng_word)>1 and len(ori_word)>1:
          data.append([eng_word,ori_word])
        eng_word=''
        ori_word=''
        flag=True
        for j in range(0,len(i)):
          try:
            if i[j][3] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ' and j==0:
              eng_word+=c_word+i[j][3:]+' '
              flag=False
          except:
            pass
          if i[j][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ' and j==1 and flag:
            eng_word+=c_word+i[j]+' '
          elif i[j][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
            eng_word+=i[j]+' '
          elif i[j][0] not in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ-':
            ori_word+=i[j]+' '
          
        continue
      if i[0][0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
        # print('data',[eng_word,ori_word])
        if len(eng_word)>1 and len(ori_word)>1:
          data.append([eng_word,ori_word])
        eng_word=''
        ori_word=''
        for index,j in enumerate(i):
          if j[0] in 'abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ':
            if j[-1]!='-' and index==0:
              eng_word=j+' '
              i_word=j+' '
              i_time=0
              continue
            elif j[-1]!='-':
              i_word+=j
              eng_word+=j+' '
            else:
              if len(i_word)>1 and len(eng_word)<1 and index==0:
                eng_word+=i_word+' '
              c_word=j[:-1]+' '
              if len(i_word)>1:
                c_word=i_word
              i_word=''
              i_time=0
              eng_word+=j+' '
          else:
            ori_word+=j+' '
      else:
        for j in i:
          ori_word+=j+' '
      i_time+=1

  if len(eng_word)>1 and len(ori_word)>1:
    # print('data',[eng_word,ori_word])
    data.append([eng_word,ori_word])
  return data

def extract_info_mode6(img_path,language,config):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h>1000:
            xl,yl,xh,yh=x,y,w,h

    print('Getting the Left and Right colunns...')
    h,w,_=image.shape
    left_block=[0,yl,xl,yl+yh]
    right_block=[xl+xh,yl,w,yl+yh]

    print('Extracting Data ...')
    l_img=image[left_block[1]:left_block[3],left_block[0]:left_block[2]]
    words=pytesseract.image_to_string(l_img,lang=language,config=config)
    word0=words.split('\n')
    output=mode6_data_extraction(word0)


    r_img=image[right_block[1]:right_block[3],right_block[0]:right_block[2]]
    words=pytesseract.image_to_string(r_img,lang=language,config=config)
    word1=words.split('\n')
    output.extend(mode6_data_extraction(word1))

    print('Extracting Complete')

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
                        # print([col1,col2,col3])
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
                    # print(word[11][0])
                    if word[11][0] in ' abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ.,\'"/' or word[11][1] in ' abcdefghijklmnopqurstuvwxyzABCDEFGHIJKLMNOPQURSTUVWXYZ.,\'"/':
                        col2+=word[11]+' '
                    else:
                        col3+=word[11]+' '
      except:
            pass
  if len(col1)>0 and len(col3)>0:
      # print([col1,col2,col3])
      result.append([col1,col2,col3])
  return result

def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    
    images_folder = os.path.join(OUTPUT_DIR, args.images_folder_name, 'images')
    
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

        if args.mode=='2':
            txt = pytesseract.image_to_string(gray_image, lang=args.language_model, config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode1(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='3':
            txt = pytesseract.image_to_data(gray_image,lang=args.language_model,config=TESSDATA_DIR_CONFIG)

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
        
    if args.mode == '1':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning','Indic Description'])
    if args.mode == '2':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning','Indic Description'])
    elif args.mode == '3':
        df = pd.DataFrame(result, columns = ['Sanskrit Word','English Vocab','Description'])
    elif args.mode == '4':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning'])
    elif args.mode == '5':
        df = pd.DataFrame(result, columns = ['English Word','Word 1','Word 2','Word 3','Word 4'])
    elif args.mode == '6':
        df = pd.DataFrame(result, columns= ['English Word','Oriya Word'])
    elif args.mode == '7':
        df = pd.DataFrame(result, columns= ['Col1','Col2','Col3'])
    
    out_file = os.path.join(OUTPUT_DIR, args.images_folder_name, 'results.csv')
    df.to_csv(out_file, index=False)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--orig_pdf_path", type=str, default=None, help="path to the input pdf file")
    parser.add_argument("-im", "--images_folder_name", type=str, default="pdf", help="type of input file | pdf/images")
    parser.add_argument("-l", "--language_model", type=str, default="Devangari", help="language to be used for OCR")
    parser.add_argument("-m", "--mode", type=str, default='1', help="mode 1 => 1 , mode 2 => 2 , Eng-Sanskrit => 3 , bhandaran => 4, English_Hindi_Tamil => 5 , English_Oriya => 6 , English_English_Hindi => 7")
    parser.add_argument("-s", "--start-page", type=str, default=None, help="Start page for OCR")
    parser.add_argument("-e", "--end-page", type=str, default=None, help="End page for OCR")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
