import os
import pickle
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load( open( 'model/rossmann_xgb_model.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( '/predict', methods=['POST'] )
def rossmann_predict():
    json_request = request.get_json()
   
    if json_request: # there is data
        if isinstance( json_request, dict ): # unique example
            df_raw = pd.DataFrame( json_request, index=[0] )
            
        else: # multiple example
            df_raw = pd.DataFrame( json_request, columns=json_request[0].keys() )
            
        pipeline = Rossmann()
        
        df1 = pipeline.data_cleaning( df_raw )
        
        df2 = pipeline.feature_engineering( df1 )
        
        df3 = pipeline.data_preparation( df2 )
        
        df_response = pipeline.get_prediction( model, df_raw, df3 )
        
        return df_response
        
    else:
        return Reponse( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )