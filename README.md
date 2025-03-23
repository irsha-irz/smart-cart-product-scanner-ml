# smart-cart-product-scanner-ml
This is application is used to train images of smart cart app product 

### Installation

Prerequisite:

a. Python 3.10.0


1. Create a virtual environment (recommended):

   - On Windows:

     ```
     python -m venv venv
     ```

   - On macOS and Linux:

     ```bash
     virtualenv venv
     ```

2. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

3. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add datasets to the dataset folder, Each product dataset should be added to a product specific folder 
![image](https://github.com/user-attachments/assets/49d6f4bb-747f-4c1d-a61c-c2a448064769)

5. Run augmentation script if you are dataset quantity is less per folder

   ```bash
   python agmentation.py
   ```
   
7. Run Pre Processing
   
   ```bash
   python preprocessing.py
   ```

8. Run training to generate model

   ```bash
   python training.py
   ```
