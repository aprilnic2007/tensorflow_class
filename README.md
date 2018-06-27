## April McKay - Final Project - Tensorflow June 2018

Input Data:
> Data for this project is based on pedestrian count data gathered from 5 Motionloft sensors in New York.
> 
> Weather data is automatically pulled from the Darksy API and available along with our count data in the API. 
> 
> Weather includes hourly temperature (F), weather conditions, and amount of precipitation (inches).


Goals:
1. Use historical pedestrian count data to train a model to learn the intracasies of this count data:
    > Seasonality
    > 
    > Weekly Patterns
    > 
    > Daily Patterns
    
2. Strech goal - Use the model to predict counts into the future based on forecast weather data (Darksky API)

– what platorm/system and installation versions you used to run your code
Mac OS X 10.10.5
ggplot                   0.11.5     
ipykernel                4.8.2      
ipython                  6.4.0      
ipython-genutils         0.2.0      
ipywidgets               7.2.1      
jupyter                  1.0.0      
jupyter-client           5.2.3      
jupyter-console          5.2.0      
jupyter-core             4.4.0      
jupyterhub               0.8.1      
Keras                    2.2.0      
Keras-Applications       1.0.2      
Keras-Preprocessing      1.0.1      
Markdown                 2.6.11     
matplotlib               2.2.2      
notebook                 5.5.0      
np-utils                 0.5.5.0    
numpy                    1.14.3     
numpydoc                 0.8.0      
pandas                   0.23.0     
pip                      10.0.1     
python-dateutil          2.7.3      
python-editor            1.0.3      
python-oauth2            1.1.0      
pytz                     2018.4      
pyzmq                    17.0.0     
requests                 2.18.4     
scikit-learn             0.19.1     
sklearn                  0.0        
spyder                   3.2.8     
tensorboard              1.8.0      
tensorflow               1.8.0      

– the sequence of how your code needs to be executed
You should be able to run the python script or jupyter notebook from top down.

This function will actually cause many of those input functions to run: 
inputs_counts(day_num,start_date,end_date) 
which is written as:
inputs_result = inputs_counts(0,'2017-11-01','2018-06-25')

The day_num input is based on this mapping: Monday = 0, Tuesday - 1. If the input is <> [0-6], all the days will be used.
