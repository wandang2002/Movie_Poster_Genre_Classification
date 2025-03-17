## Movie Poster Genre Classification

Disclaimer: This is a club project (with the initial idea coming from my great friend Nathaniel). I decided to improve this project beyond just school and club activities, turning it into a practical learning opportunity for data cleaning and machine learning.

### Requirement
Python : version 3.9 - 3.12 (in order to match with Tensorflow). I'm using Python 3.11

### Instruction
1. Create a virtual environment with Python 3.11

> **Bash**  
> ```bash
> py -3.11 -m venv tensorflow_env
> ```

This tells the launcher to use Python 3.10 to create a new virtual environment named `tensorflow_env`.

2. Active the virtual environment

> **Bash**  
> ```bash
> tensorflow_env\Scripts\activate
> ```

(Adjust the path if you created `tensorflow_env`` in a different location.)

You should now see `(tensorflow_env)` at the beginning of your terminal prompt. Any `python` or `pip` commands in this prompt will use Python 3.11 from your newly created environment.

3. Install requirements.txt

> **Bash**  
> ```bash
> pip install -r requirements.txt
> ```

4. Data preparation

> **Bash**  
> ```bash
> python data_cleaning.py
> ```

This will clean and binarize genre labels to support multi-label classification, ensuring the dataset was properly structured for model training and evaluation.

> **Bash**  
> ```bash
> python download_images.py
> ```

This will download the images locally to the machine so that the model can train faster without needing to access the URL.

5. **Training the model**

> **Bash**
> ```bash
> python movie_genre_classification.py
> ```

At this point, just go and get some coffee, do another works, and get back in about an hour (or slower, depend on your computer)

6. **Run the model**
> **Bash**
> ```bash 
> cd movie_poster_app
> python app.py
> ```
