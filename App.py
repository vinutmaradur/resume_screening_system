import streamlit as st
import pandas as pd
import base64,random
import time,datetime
from io import BytesIO

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image

from pyresparser import ResumeParser
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter

from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
from streamlit_tags import st_tags
import io , os
import cv2
import pdf2image
import plotly.express as px

import tensorflow as tf
Sequential = tf.keras.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
load_model = tf.keras.models.load_model

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import mysql.connector

from yt_dlp import YoutubeDL

import spacy
nlp = spacy.load("en_core_web_sm")
def extract_name_from_text(text):
    lines = text.split('\n')
    for line in lines[:5]:  
        line = line.strip()
        if not line or any(char.isdigit() for char in line) or '@' in line:
            continue
        if len(line.split()) <= 4:  
            return line.title()  

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.title()

    return None

def fetch_yt_video(link):
    ydl_opts = {'quiet': True, 'skip_download': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        return info.get('title', 'Video Title Not Found')

def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):    
    if isinstance(file,str):
        with open(file,"rb") as f:
            file = io.BytesIO(f.read())
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    file_stream = io.BytesIO(file.read())
    with file_stream as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    
    return text.encode("utf-8","ignore").decode("utf-8")

def extract_images_from_pdf(pdf_path, output_folder):
    images = pdf2image.convert_from_path(pdf_path)
    image_paths = []
    os.makedirs(output_folder, exist_ok=True)
    
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)
    
    return image_paths

def extract_faces(image_path, model=None, class_labels=None):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (80, 80)) / 255.0
        face_array = np.expand_dims(face_resized, axis=0)

        prediction = None
        if model:
            pred = model.predict(face_array)
            pred_class = np.argmax(pred, axis=1)[0]
            prediction = class_labels[pred_class] if class_labels else f"Class {pred_class}"

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        return face_pil, prediction

    return None, None


def create_folder():
    folder_name = "Job_Description"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# Load a pre-trained model or define a simple CNN
def build_cnn_model(input_shape=(80, 80, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = build_cnn_model()

def save_job_description(file, folder):
    file_path = os.path.join(folder, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def read_text(file_path):
    encodings = ["utf-8","ISO-8859-1","windows-1252"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return "Error:Unable to decode file content."
    

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Database Connection
def create_connection():
    return mysql.connector.connect(
        host="localhost",      # Change if needed (e.g., remote server)
        user="root",  # Replace with your MySQL username
        password="Vinut@2001",  # Replace with your MySQL password
        database="resume_database"   # Ensure you have created this database
    )
conn = create_connection()
cursor = conn.cursor()

st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='./Logo/logo6.png',
)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

def run():
    img = Image.open('./Logo/logo4.jpg')
    img = img.resize((800,250))
    st.image(img)
    st.title("AI Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '[©Developed by Vinut](https://www.linkedin.com/in/vinut-maradur/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    
    # Create table
    def create_table():
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resume (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            mobile_number VARCHAR(20),
            no_of_pages INT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            Predicted_Field VARCHAR(255),
            User_level VARCHAR(255),
            skills TEXT,
            recommended_skills TEXT,
            courses TEXT
            );
        '''
        )
        conn.commit()
        conn.close()

    # Call this function at the start
    create_table()

    def save_resume_to_db(resume_data,predicted_field='',user_level=''):
        conn = create_connection()
        cursor = conn.cursor()
        values = (
            resume_data.get('name', 'Unknown'),
            resume_data.get('email', 'N/A'),
            resume_data.get('mobile_number', 'N/A'),
            resume_data.get('no_of_pages', 0),
            predicted_field,
            user_level,
            ', '.join(resume_data.get('skills',[])),
            ', '.join(recommended_skills),
            ', '.join(rec_course)
        )
        sql = '''INSERT INTO resume (name, email, mobile_number, no_of_pages, Predicted_Field, User_level, skills, recommended_skills, courses)
         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
        cursor.execute(sql, values)
        conn.commit()
        conn.close()
    
    if choice == 'User':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''',
                    unsafe_allow_html=True)
        job_desc_file = st.file_uploader("Upload Job Description (TXT only)", type=["docx"])
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        

        if pdf_file is not None:
            with st.spinner('Uploading your Resume...'):
                time.sleep(4)
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            with open(save_image_path, "rb") as f:
                resume_text = pdf_reader(f)

            resume_data = ResumeParser(save_image_path).get_extracted_data()

            if resume_data and job_desc_file:
                folder = create_folder()
                jd_path = save_job_description(job_desc_file, folder)
                job_description = read_text(jd_path)
            
        
                similarity = calculate_similarity(resume_text, job_description)
                st.subheader("Match Results")
                st.write(f"Your resume matches the job description by: {similarity:.2f}%")

            if resume_data:
                ## Get the whole resume data
                resume_text = pdf_reader(save_image_path)
                resume_data = ResumeParser(save_image_path).get_extracted_data()

                extracted_name = extract_name_from_text(resume_text)
                resume_data['name'] = extracted_name if extracted_name else resume_data.get('name', 'Unknown')

                st.header("**Resume Analysis**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info**")
            try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
            except:
                    pass
            cand_level = ''
            if resume_data['no_of_pages'] == 1:
                cand_level = "Fresher"
                st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
            elif resume_data['no_of_pages'] == 2:
                cand_level = "Intermediate"
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
            elif resume_data['no_of_pages'] >=3:
                cand_level = "Experienced"
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)

            
            # st.subheader("**Skills Recommendation💡**")
            ## Skill shows
            keywords = st_tags(label='### Your Current Skills',
            text='See our skills recommendation below',
            value=resume_data['skills'],key = '1  ')

            ##  keywords
            ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep Learning','flask','streamlit']
            web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress',
                               'javascript', 'angular js', 'c#', 'flask']
            android_keyword = ['android','android development','flutter','kotlin','xml','kivy']
            ios_keyword = ['ios','ios development','swift','cocoa','cocoa touch','xcode']
            uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','adobe after effects','after effects','adobe premier pro','premier pro','adobe indesign','indesign','wireframe','solid','grasp','user research','user experience']

            recommended_skills = []
            reco_field = ''
            rec_course = ''
            ## Courses recommendation
            for i in resume_data['skills']:
                    ## Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',"Flask",'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '2')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        break

                    ## Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress','Javascript','Angular JS','c#','Flask','SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '3')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)
                        break

                    ## Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '4')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        break

                    ## IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '5')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        break

                    ## Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                        text='Recommended skills generated from System',value=recommended_skills,key = '6')
                        st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        break
            
            # Extract and display faces
            extracted_images = extract_images_from_pdf(save_image_path, "./Extracted_Images")
            st.subheader("Extracted Faces")
            face = False
            for img_path in extracted_images:
                face, label = extract_faces(img_path, model=cnn_model, class_labels=["Happy", "Neutral", "Serious"])
                if face is not None:
                    #st.image(face, caption=f"Extracted Face from {os.path.basename(img_path)}", width=80)
                    st.image(face, caption=f"Prediction: {label}", width=100)
                    break
                
            if face is None:
                    st.warning("No faces detected in the document.")
            
            ### Resume writing recommendation
            st.subheader("**Resume Tips & Ideas💡**")
            resume_score = 0
            
            # Objective / Summary
            if 'Objective' in resume_text or 'Summary' in resume_text:
                resume_score += 10
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Great! You've added an Objective or Summary.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Please add a career Objective or Summary to highlight your goals.</h5>", unsafe_allow_html=True)

            # Contact Information
            if any(x in resume_text for x in ['Email', '@', 'Phone', 'Contact']):
                resume_score += 10
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Great! Contact Information is present.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Please add your Contact Information so recruiters can reach you.</h5>", unsafe_allow_html=True)

            # Education
            if 'Education' in resume_text or 'Degree' in resume_text:
                resume_score += 10
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Excellent! Education section is included.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Include your Education background to show your qualifications.</h5>", unsafe_allow_html=True)

            # Experience / Internship
            if 'Experience' in resume_text or 'Internship' in resume_text:
                resume_score += 20
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Fantastic! Work Experience or Internships are included.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Add Work Experience or Internships to highlight your practical exposure.</h5>", unsafe_allow_html=True)

            # Projects
            if 'Projects' in resume_text:
                resume_score += 10
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Good job! Projects are listed.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Include Projects to demonstrate applied knowledge and hands-on skills.</h5>", unsafe_allow_html=True)

            # Skills / Technologies
            if 'Skills' in resume_text or 'Technical Skills' in resume_text:
                resume_score += 10
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Skills section is present. Well done!</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Include Skills to show what tools and technologies you know.</h5>", unsafe_allow_html=True)

            # Achievements
            if 'Achievements' in resume_text or 'Awards' in resume_text:
                resume_score += 10
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You’ve listed your Achievements.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Mention Achievements or Awards to stand out from other candidates.</h5>", unsafe_allow_html=True)

            # Certifications / Courses
            if 'Certifications' in resume_text or 'Courses' in resume_text:
                resume_score += 5
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Great! Certifications or Courses are included.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Add Certifications or Courses to highlight continued learning.</h5>", unsafe_allow_html=True)

            # Hobbies / Interests
            if 'Hobbies' in resume_text or 'Interests' in resume_text:
                resume_score += 5
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Hobbies or Interests are included. Nice touch!</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Consider adding Hobbies or Interests to show personality.</h5>", unsafe_allow_html=True)

            # Declaration
            if 'Declaration' in resume_text:
                resume_score += 5
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] Declaration section is present.</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Add a Declaration to confirm the authenticity of your resume.</h5>", unsafe_allow_html=True)

            # References / Links
            if 'References' in resume_text or 'LinkedIn' in resume_text or 'GitHub' in resume_text:
                resume_score += 5
                st.markdown("<h5 style='text-align: left; color: #1ed760;'>[+] You've included References or Professional Links. Good work!</h5>", unsafe_allow_html=True)
            else:st.markdown("<h5 style='text-align: left; color: #FF0000;'>[-] Include References or LinkedIn/GitHub profile links to boost credibility.</h5>", unsafe_allow_html=True)

            st.subheader("**Resume Score📝**")
            st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )
            my_bar = st.progress(0)
            score = 0
            for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
            st.success('** Your Resume Writing Score: ' + str(score)+'**')
            st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")
            st.balloons()

            ## Resume writing video
            st.header("**Bonus Video for Resume Writing Tips💡**")
            resume_vid = random.choice(resume_videos)
            res_vid_title = fetch_yt_video(resume_vid)
            st.subheader("✅ **"+res_vid_title+"**")
            st.video(resume_vid)

            ## Interview Preparation Video
            st.header("**Bonus Video for Interview Tips💡**")
            interview_vid = random.choice(interview_videos)
            int_vid_title = fetch_yt_video(interview_vid)
            st.subheader("✅ **" + int_vid_title + "**")
            st.video(interview_vid)

            with open(save_image_path, "rb") as f:
                st.download_button(
                      label="📥 Download Your Resume",
                      data=f,
                      file_name=pdf_file.name,
                      mime="application/pdf"
                   )

            save_resume_to_db(resume_data,reco_field,cand_level)
            st.success("Resume saved to the database!")
        else:
            st.warning('Upload your Resumes and Job Description Here')
    
    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        # st.sidebar.subheader('**ID / Password Required!**')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'vinut' and ad_password == 'vinut123':
                st.success("Welcome Mr vinut !")
                # Display Data
                cursor.execute('''SELECT * FROM resume''')
                data = cursor.fetchall()
                st.header("**User's Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'mobile_number', 'Total Page','upload_time','Predicted_Field', 'User_level','skills','recommended_skills','courses'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df,'resume.csv','Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                query = 'select * from resume;'
                conn = create_connection() 
                plot_data = pd.read_sql(query, conn)

                ## Pie chart for predicted field recommendations
                if not plot_data.empty and 'Predicted_Field' in plot_data.columns:
                    field_counts = plot_data['Predicted_Field'].value_counts().reset_index()
                    field_counts.columns = ['Predicted_Field', 'Count']
                    if not field_counts.empty:
                        st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                        fig = px.pie(field_counts, values='Count', names='Predicted_Field',
                            title='Predicted Field according to the Skills')
                        st.plotly_chart(fig)
                else:
                    st.info("No predicted field data available to plot.")

                ### Pie chart for User's👨‍💻 Experienced Level
                if not plot_data.empty and 'User_level' in plot_data.columns:
                    field_counts = plot_data['User_level'].value_counts().reset_index()
                    field_counts.columns = ['User_level','count']
                    if not field_counts.empty:
                        st.subheader("**pie-Chart for User's Experienced Level**")
                        fig = px.pie(field_counts, values='count', names='User_level',title='Pie-Chart📈 for Users👨‍💻 Experienced Level')
                        st.plotly_chart(fig)
                else:
                    st.info("No user level field data available to plot.")

            else:
                st.error("Wrong ID & Password Provided")
run()
