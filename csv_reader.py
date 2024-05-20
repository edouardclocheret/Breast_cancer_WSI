import csv


def is_sane(patient):
    file_path = "C:\Edouard\Tecnico\Code\Train_features\stage_labels.csv"
    k=1
    return patient in select_patient_k_node_unsane(file_path, k)

def set_of_sanes():
    file_path = "C:\Edouard\Tecnico\Code\Train_features\stage_labels.csv"
    k=1
    return select_patient_k_node_unsane(file_path, k)

def select_patient_k_node_unsane(file_path, k) :
    # Returns set()  where( count(unsane nodes) < k ) 
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file : 
        reader =  csv.DictReader(file) 
        res = set()

        patient_number =-1

        line_count =1
        pose_node_count = 0
        for row in reader :

            if patient_number >=0 : 
                
                
                if (line_count-1)%6>0 :

                    if row['stage']!= 'negative': 
                        pose_node_count+=1

                    if  (line_count-1)%6 ==5:
                        
                        if pose_node_count < k :
                            res.add(patient_number)
                            
                        
                        pose_node_count =0
            
            line_count +=1
            patient_number = (line_count-1) //6

        
    return res



def main (): 
    file_path = "C:\Edouard\Tecnico\Code\Train_features\stage_labels.csv"
    k = 2
    print(f"Patients having exactly {k} infected nodes\n")
    print(select_patient_k_node_unsane(file_path, k+1)-select_patient_k_node_unsane(file_path, k))
        
    print(f"set of sanes \n{set_of_sanes()}")


if __name__ == "__main__":
    main()