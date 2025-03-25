# **📄 ResuMate - AI Resume Evaluator**  

**ResuMate** is an AI-powered resume analysis tool designed to help job seekers optimize their resumes for applicant tracking systems (ATS) and job descriptions. It evaluates resumes based on similarity to job descriptions, ATS readability, grammar errors, missing keywords, action verbs, and more.  

---

## **🚀 Features**  

- ✅ **Extract Text from PDF**: Reads and analyzes resumes and job descriptions from PDFs.  
- ✅ **Keyword Extraction**: Identifies key terms in job descriptions and resumes.  
- ✅ **Resume vs Job Description Similarity**: Computes a similarity score based on **TF-IDF** and **cosine similarity**.  
- ✅ **ATS Readability Score**: Evaluates how easy a resume is for ATS systems to parse.  
- ✅ **Grammar Check**: Identifies grammatical errors using **LanguageTool**.  
- ✅ **Missing Keywords Detection**: Highlights important keywords missing from resumes.  
- ✅ **Action Verbs Count**: Analyzes the use of impactful action verbs.  
- ✅ **Resume Section Detection**: Checks if critical sections (Education, Skills, Experience, Projects, Certifications) are present.  
- ✅ **Word Count Analysis**: Counts total words in the resume.  
- ✅ **Graphical Report**: Generates a bar chart comparing resumes on similarity and ATS scores.  
- ✅ **Downloadable Report**: Creates a **PDF report** with evaluation results and graphs.  

---

## **📌 Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/ResuMate.git
cd ResuMate
```

### **2️⃣ Set Up a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## **📂 File Structure**  
```
ResuMate/
│── main.py                  # Streamlit app file
│── requirements.txt          # Required dependencies
│── README.md                 # Documentation
│── plot.png                  # Saved graph (temporary)
│── .gitignore                # Ignoring unnecessary files
```

---

## **▶️ Usage**  

### **Run the App**  
```bash
streamlit run main.py
```

### **Upload the following files in the UI:**  
- **Job Description (PDF)**
- **Resume(s) (PDFs)**  

Then, click **🔍 Evaluate Resumes** to generate results.

---

## **📊 Output Results**  
Each resume is evaluated based on:  
- **📊 Similarity Score (%)** – Measures how well the resume matches the job description.  
- **📖 ATS Readability Score** – Evaluates the ease of parsing.  
- **🔠 Grammar Errors** – Identifies grammatical mistakes.  
- **💼 Action Verbs Used** – Counts powerful action verbs.  
- **📜 Word Count** – Displays total word count.  
- **🔍 Missing Keywords** – Shows important missing terms.  
- **📑 Resume Sections** – Checks if key sections are present.  
- **📊 Graphical Analysis** – Displays a bar chart comparing resumes.  
- **📥 PDF Report Download** – Generates a detailed evaluation report with findings & graphs.

---

## **📦 Dependencies**  
- **Python 3.8+**  
- **Streamlit**  
- **PyPDF2**  
- **Matplotlib**  
- **NumPy**  
- **Pandas**  
- **TextStat**  
- **LanguageTool-Python**  
- **Scikit-Learn**  
- **ReportLab**  

---

## **⚒️ Install Dependencies**
```bash
pip install -r requirements.txt
```

## **🛠️ Contributing**  
1. Fork the repository  
2. Create a new branch (`feature-branch`)  
3. Commit your changes  
4. Push to your branch  
5. Open a pull request  

---

## **📜 License**  
This project is open-source and available under the **MIT License**.

---

## **📩 Contact**  
For questions or suggestions, feel free to reach out via [My Email](sj6740@srmist.edu.in).  

## **MIT License**

Copyright (c) 2025 Sam T James  

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"),  
to deal in the Software without restriction, including without limitation  
the rights to use, copy, modify, merge, publish, distribute, sublicense,  
and/or sell copies of the Software, and to permit persons to whom the Software  
is furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,  
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A  
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT  
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION  
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE  
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


🚀 **Happy Resume Optimization!** 🚀
