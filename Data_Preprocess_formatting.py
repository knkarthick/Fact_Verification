# Reformatting the Original Data
import os
# os.chdir(r"C:\Users\91720\Documents\Fact_Verification")
import json
import pandas as pd

def preprocess(fname, outname):
    if outname != 'test':
        data_list = []
        with open(fname, 'rb') as f:
            for line in f:
                data = json.loads(line)
        
                print(data.keys())
                
                print(data['id'], data['rumor'], data['label'])
        
                rumor_id, rumor, label = data['id'], data['rumor'], data['label']
                
                evidence_list = [evidence for evidence in data['evidence']]
                
                evidence_ids = [evidence[1] for evidence in evidence_list]
                
                timeline_list = [timeline for timeline in data['timeline']]
                for timeline in timeline_list:
                    is_evidence = "1" if timeline[1] in evidence_ids else "0"
                    data_list.append([rumor_id, rumor, label, is_evidence, timeline[0].split("/")[-1], timeline[1], timeline[2]])
            
        df = pd.DataFrame(data_list, columns = ['id', 'rumor', 'label', 'is_evidence', 'username', 'timeline_id', 'timeline'])
        df.to_csv("./data/data_intermediate/"+outname+".csv", index=False)
        
        data_list = []
        with open(fname, 'rb') as f:
            for line in f:
                data = json.loads(line)

                print(data.keys())

                print(data['id'], data['rumor'], data['label'])

                rumor_id, rumor, label = data['id'], data['rumor'], data['label']

                evidence_list = [evidence for evidence in data['evidence']]

                evidence_ids = [evidence[1] for evidence in evidence_list]

                evidence = " ".join([evidence[2] for evidence in evidence_list])

                username = [evidence[0].split("/")[-1] for evidence in evidence_list]

                data_list.append([rumor_id, rumor, label, evidence_ids, evidence, username])

        df = pd.DataFrame(data_list, columns = ['id', 'rumor', 'label', 'evidence_id', 'evidence', 'username'])
        df.to_csv("./data/data_intermediate/"+outname+"_evidence_class.csv", index=False)
        print("Processing Done!")
    else:
        data_list = []
        with open(fname, 'rb') as f:
            for line in f:
                data = json.loads(line)
        
                print(data.keys())
                
                print(data['id'], data['rumor'])
        
                rumor_id, rumor = data['id'], data['rumor']
                
                timeline_list = [timeline for timeline in data['timeline']]
                for timeline in timeline_list:
                    data_list.append([rumor_id, rumor, timeline[0].split("/")[-1], timeline[1], timeline[2]])
            
        df = pd.DataFrame(data_list, columns = ['id', 'rumor', 'username', 'timeline_id', 'timeline'])
        df.to_csv("./data/"+outname+".csv", index=False)
        
        with open(fname, 'rb') as f:
            for line in f:
                data = json.loads(line)
        df = pd.DataFrame(data)
        df.to_csv("./data/"+outname+"_original.csv", index=False)
        
        print("Processing Done!")

if __name__ == "__main__":
    print("Processing Training Data")
    preprocess('./data/original_data/English_train.json', 'train')
    
    print("Processing Dev Data")
    preprocess('./data/original_data/English_dev.json', 'dev')
    
    print("Processing Test Data")
    preprocess('./data/original_data/English_test.json', 'test') 