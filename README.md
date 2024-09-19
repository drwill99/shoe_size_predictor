# Shoe Size Prediction Program

## Author
**Dallin Williams**

## Function of the Program
This program utilizes a Linear Regression model to predict shoe size based on a user's height and sex. The user can input their height in feet and inches, and the program will provide a prediction of their shoe size. The program also allows users to check the accuracy of the prediction against their actual shoe size and plot their shoe size on a scatterplot along with other entries from the dataset, filtered by matching sex.

## How the Program Works
This Python application uses various libraries to accomplish the task of shoe size prediction:

- **pandas**: This library is used to load, clean, and manipulate the dataset (`shoe-size-samples.csv`). It filters invalid data points, imputes missing values, and organizes the dataset for analysis.
  
- **scikit-learn**: The Linear Regression model from this machine learning library is employed to learn the relationship between height, sex, and shoe size. The model is trained on the cleaned dataset and used to predict shoe sizes based on the user's input.

- **matplotlib**: This library is used to create visualizations. The scatterplot function helps visualize the predicted shoe size in comparison with other entries in the dataset that match the user’s sex.

- **tkinter**: The tkinter library is used to create the Graphical User Interface (GUI) for the program. This provides an easy-to-use interface where users can input their height, sex, and actual shoe size, and interact with the program through buttons.

### Concepts Utilized:
- **Linear Regression**: This is a machine learning algorithm that models the relationship between two or more variables by fitting a linear equation to observed data. In this case, it is used to predict shoe size based on height and sex.
  
- **Data Imputation**: The program fills missing values for height in the dataset using the median height to maintain data integrity and ensure the model can be trained effectively.
  
- **Data Filtering**: Entries in the dataset that have unrealistic shoe sizes (less than 30 or greater than 60) or heights (less than 100 cm or greater than 250 cm) are filtered out to improve prediction accuracy.

## How to Use the Program
### Step-by-Step Instructions:

1. **Download the Required Files**:
   - Obtain the `shoe-size-samples.csv` dataset, which contains shoe sizes, heights, and sex information for various individuals.
   - Download the Python script file, `shoe_size_prediction.py`.

2. **Install Necessary Libraries**:
   Ensure the following Python libraries are installed:
   - `pandas`
   - `scikit-learn`
   - `matplotlib`
   - `tkinter` (usually built into Python)

   You can install the required libraries using the following `pip` commands:
   ```bash
   pip install pandas scikit-learn matplotlib

3. **Run the Program**:
    - Open a terminal (or command prompt) and navigate to the directory containing the Python script and the CSV file.
    - Execute the Python script with the following command: `python shoe_size_prediction.py`

4. **Interacting with the Program**:
    - A GUI window will open, allowing you to input your height (in feet and inches) and select your sex (`m` for male, `f` for female).
    - Click the "Predict Shoe Size" button to generate a prediction based on the input data.
    - To check the accuracy of the prediction, input your actual shoe size and click "Check Prediction Accuracy." The program will compare the predicted and actual sizes and display the accuracy.
    - To visualize your shoe size on a scatterplot along with other entries from the dataset (filtered by matching sex), click the "Plot My Shoe Size" button.
    - Use the "Reset" button to clear all input fields and start over.

## Program Limitations, Real-World Applications, and Future Improvements
### Limitations:
-  **Limited Features**: The model only considers height and sex as factors for predicting shoe size. Other relevant factors, such as age, weight, or foot width, are not included, which may limit prediction accuracy.

- **Dataset Size**: The current dataset may not contain enough examples to generalize well, especially for individuals who fall outside common height or shoe size ranges. More data would improve the model’s ability to make accurate predictions for a wider range of individuals.

- **Model Simplicity**: Linear Regression is a basic machine learning model that assumes a linear relationship between variables. More complex relationships may not be captured with this approach.

### Real-World Applications:
- **Teaching Aid**: Most realistically, this program and dataset can be used as an example of the Linear Regression model in basic machine learning courses at the high school or undergraduate levels.

- **E-commerce**: This program can be used by online shoe retailers to help customers determine their shoe size based on simple inputs like height and sex. It could enhance the user experience by offering personalized size recommendations.

- **Sportswear Industry**: The program could be integrated into the footwear design and sales process, especially for athletic shoe brands that want to offer custom shoe sizes tailored to users' needs.

- **Retail Assistants**: In physical stores, this type of program could be used as a tool by sales assistants to quickly suggest shoe sizes for customers without needing detailed foot measurements.

### Future Improvements:
- **Add More Detail Code Comments**: This would be to aid teachers and students in better following along with the program and exactly what each module and function does.

- **Incorporate Additional Features**: Adding features like foot width, age, or weight could improve the accuracy of predictions by considering other relevant factors.

- **Larger and More Diverse Dataset**: Expanding the dataset to include more entries from different demographics would make the model more reliable, especially for underrepresented shoe sizes and height ranges.

- **Advanced Models**: The current implementation uses a simple Linear Regression model. Future versions of the program could explore more sophisticated machine learning algorithms such as Decision Trees, Random Forests, or Neural Networks, which may improve prediction accuracy.

- **Enhanced User Interface**: Future updates could improve the GUI by adding drop-down menus for height selection, better visualization options, and more interactive features to improve the user experience.

## Contact Me
### Repo Specific Discussion Board
- https://github.com/drwill99/shoe_size_predictor/discussions/1
