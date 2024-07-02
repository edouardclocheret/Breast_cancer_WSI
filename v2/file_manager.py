import numpy as np
import pandas as pd
import h5py
import os
import openpyxl

def assign_paths(computer):
    
    if computer == 'GIGA-CHAD':
        features_dir = "C:\\Edouard\\Tecnico\\Code\\Train_features"
        path_to_csv = "C:\\Edouard\\Tecnico\\Code\\Train_features\\stage_labels.csv"
        matrix_file = "C:\Edouard\TECNICO\Code\outputs\hist\\"
        excel_metrics = "C:\Edouard\TECNICO\Code\outputs\excel_metrics.xlsx"

    else:
        features_dir = "../../Data/RitaP/Camelyon17/FEATURES_DIRECTORY/h5_files/"
        path_to_csv = "../../Data/RitaP/Camelyon17/stage_labels.csv"
        matrix_file = "../Edouard/outputs"
        excel_metrics = "../Edouard/outputs"
    return features_dir, path_to_csv, matrix_file, excel_metrics

def load_data(centers_vect, features_dir):
    #outputs :
    #n_points_vect is the number of patches per image
    #patient_id_vect is the patient id per patch (to avoid errors if one node is missing)
    all_features = []
    all_coords = []
    n_points_vect =[]
    patient_id_vect =[]

    for center in centers_vect:
        for i in range(center*20, (center+1)*20):
            print(f"Loading patient {i}")
            for j in range(0, 5):
                filename = os.path.join(features_dir, f"center{center}", f"patient_{i:03d}_node_{j}.h5")
                if os.path.exists(filename):
                    with h5py.File(filename, 'r') as f:
                        coords = f['coords'][:]
                        features = f['features'][:]
                        all_coords.append(coords)
                        all_features.append(features)
                        n_points_vect.append(coords.shape[0])
                else:
                    print(f"There is no patient {i}, node {j}")
                    exit(0)
                patient_id_vect.append(i)

    return np.concatenate(all_coords), np.concatenate(all_features, axis=0), n_points_vect, patient_id_vect


def read_csv_to_matrix(csv_file):
    
    df = pd.read_csv(csv_file)

    matrix = np.full((100, 5), "NO_DATA_")
    for _, row in df.iterrows():
        file_name = row['patient']
        stage = row['stage']
        
        if 'node' in file_name:
            patient_num = int(file_name.split('_')[1])
            node_num = int(file_name.split('_')[-1].split('.')[0])
            matrix[patient_num, node_num] = stage
    
    return matrix

def load_binary_labels(path_to_csv, centers, exclude_pairs=None):

    matrix = read_csv_to_matrix(path_to_csv)
    
    binary_labels = []
    
    for center in centers:
        start_patient = center * 20
        end_patient = (center+1) * 20
        
        for patient in range(start_patient, end_patient):
            for node in range(5):
                if exclude_pairs and (patient, node) in exclude_pairs:
                    
                    continue
                binary_labels.append(matrix[patient,node]=='negative')
    y = np.array(binary_labels)
    
    return y

def load_multi_labels(path_to_csv, centers, exclude_pairs=None):
    matrix = read_csv_to_matrix(path_to_csv)
    
    labels = []
    
    for center in centers:
        start_patient = center * 20
        end_patient = (center+1) * 20
        
        for patient in range(start_patient, end_patient):
            for node in range(5):
                if exclude_pairs and (patient, node) in exclude_pairs:
                    
                    continue
                labels.append(matrix[patient,node])
    y = np.array(labels)
    
    return y


def save_hist(matrix, hist_dir, centers):
    base_filename = hist_dir+"_centers_"+str(centers)+'.h5'

    # Check if the file already exists and modify the filename if necessary
    filename = base_filename
    while os.path.exists(base_filename):
        filename = base_filename.replace('.h5', '_bis.h5')
        base_filename = filename

    # Save the array to an HDF5 file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('hist_matrix', data=matrix)

def save_metrics(accuracy, class_report, conf_matrix, file_path):
    class_report_df = pd.DataFrame(class_report).transpose()
    conf_matrix_df = pd.DataFrame(conf_matrix)

    metrics_data = {
        'Metric': ['Accuracy', 'Classification Report', 'Confusion Matrix'],
        'Value': [accuracy, class_report_df, conf_matrix_df]
    }

    try:
        book = openpyxl.load_workbook(file_path)
        writer = pd.ExcelWriter(file_path, engine='openpyxl')
        writer.book = book
        start_row = writer.sheets['Sheet1'].max_row + 2  # Find the starting row after existing data + 1 blank row
    except FileNotFoundError:
        writer = pd.ExcelWriter(file_path, engine='openpyxl')
        start_row = 0

    # Step 4: Write the metrics to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        if start_row > 0:
            # Write a blank row and title row
            blank_df = pd.DataFrame([''], columns=[''])
            blank_df.to_excel(writer, startrow=start_row-1, index=False, header=False)
        
        # Write title row
        title_df = pd.DataFrame(['Accuracy, Classification Report, Confusion Matrix'], columns=[''])
        title_df.to_excel(writer, startrow=start_row, index=False, header=False)
        
        # Write accuracy
        accuracy_df = pd.DataFrame([['Accuracy', accuracy]])
        accuracy_df.to_excel(writer, startrow=start_row+1, index=False, header=False)

        # Write classification report
        class_report_df.to_excel(writer, startrow=start_row+2, sheet_name='Sheet1')
        
        # Write confusion matrix
        conf_matrix_df.to_excel(writer, startrow=start_row+len(class_report_df)+3, sheet_name='Sheet1')

    writer.save()

def main():
    csv_file_path = "C:\\Edouard\\Tecnico\\Code\\Train_features\\stage_labels.csv"
    matrix = read_csv_to_matrix(csv_file_path)
    print(matrix)
    centers = [0,1,2,3,4]
    
    y = load_binary_labels(csv_file_path, centers, exclude_pairs=[(35,2)])
    print(y.shape)
    return 0

if __name__ == '__main__':
    main()