# HKUST-COMP4901K-Project3
 Language Model Course Project  
### Train a model:  
python main.py -mode train -saved_model models/model.h5 -student_id 12345678 -epochs 1 -batch_size 32 -embedding_dim 100 -hidden_size 500 -drop 0.5  

### Generate prediction for valid.csv:  
python main.py -mode test -saved_model models/model.h5 -input data/valid.csv -student_id 12345678  

### Evaluate performance of your model:  
python scorer.py -submission 12345678_valid_result.csv  

### Once you have finished tuning your model, you can make a submission for test.csv:  
python main.py -mode test -saved_model models/model.h5 -input data/test.csv -student_id 12345678
