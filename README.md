Fact_Verification


Steps to Run:
1. python Data_Preprocess_formatting.py to Format the data 
2. python Data_Preprocess_Cleaning.py to clean the data
3. Run the notebook ./data_augmentation/Data_Augmentation_From_Bert.ipynb in environment (pip install -r requirements_pytorch.txt) to augment the data 
4. Run the notebook BERT_Evidence_Classification_FINAL.ipynb to train Model - I 
5. Run the notebook BERT_Evidence_Classification_FINAL.ipynb to train Model - I 
6. Run the inference notebook Inference_Pipeline.ipynb with inputs from step 4 & 5
	model1_path = final model I path = './models/evidence_identification_model/model.epoch00-loss1.22'
	model2_path = final model II path = './models/evidence_classification_model/model.epoch02-loss0.93'
	
	threshold1 = probability threshold to consider for model I = 0.1
	threshold2 = probability threshold to consider for model II = 0.282
7. Use the two csv files generated at ./results/ for submission
	Evidence_retrieval_output = Rumor with supporting evidence and probabilities
	Rumor_Verification_output = Rumor with top evidences to either substantiate SUPPORT/ REFUTE/ NOT ENOUGH INFO.