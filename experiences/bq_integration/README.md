#BigQuery to TF example
This example proposes an integration of information from BigQuery to train a model using TensorFlow and Keras.

For this integration example was used this module of [pandas](https://pandas.pydata.org/):
- pandas-gbq, [more information](https://pandas-gbq.readthedocs.io/en/latest/) 

##Connect to Google Cloud
for this operation it is recommended to use a service access.

The example uses:
- ConfigParser: methodology to extract the necessary information from a configuration file.

file format: config.env

format:
``` 
[google]
cloud_id=projectid
service_key=servicekey.json 
```
>Note: remember to create the configuration file and update these values to run the example.

reading the configuration:
```
config.read('config.env')
```
Configuring Cloud Project:
```project_id = config.get('google','cloud_id')```

Configuring service-key for the Cloud Project:

```df_train = pd.io.gbq.read_gbq('''SELECT * FROM [socialagilelearning:iris.training]''', project_id=project_id, private_key=config.get('google','service_key'), verbose=False)```

>Note: private_key