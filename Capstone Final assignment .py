#!/usr/bin/env python
# coding: utf-8

# ## Discovering the best location for opening a new restaurant in Hong Kong Island

# #### Importing all the required libraries

# In[1]:


import requests # library to handle requests
import pandas as pd # library for data analysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation
 
 
get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
 
# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize
 
 
get_ipython().system(' pip install folium==0.5.0')
import folium # plotting library
 
print('Folium installed')
print('Libraries imported.')


# #### List out all the parameters required for FourSquare API

# In[2]:


CLIENT_ID = 'SIB1HHX3KY0MKB1MLAW0O1ATKYMWIEFRBNYABIC2WXJCY5RV' # your Foursquare ID
CLIENT_SECRET = 'EJZDHTTMN22S5EOSZYNXCCCETEEIXKKIKUVX3HZZR2QKAMJE' # your Foursquare Secret
ACCESS_TOKEN = 'NFL0J2F51ZXDGOXWCY0CKOVSNOSCTD00ZFJUHY3JDILAJF4Y' # your FourSquare Access Token
VERSION = '20180604'
LIMIT = 100
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# #### Find out the latitude and longitude of Hong Kong Island

# In[3]:


city = 'Hong Kong Island, Hong Kong'

geolocator = Nominatim(user_agent='foursquare_agent')
location = geolocator.geocode(city)
hkI_latitude = location.latitude
hkI_longitude = location.longitude
print ('The latitude and longitude of Hong Kong Island are:', hkI_latitude, 'and', hkI_longitude)


# #### Import a Hong Kong district dataframe obtained from wikipedia

# In[4]:


hkdistrict18 = "https://en.wikipedia.org/wiki/Districts_of_Hong_Kong"

hkdistrict18_df=pd.read_html(hkdistrict18)
len(hkdistrict18_df)


# In[5]:


hkdistrict18_df[6] #find out the table that show all the districts & region of Hong Kong


# In[6]:


hkdistrict18_data=hkdistrict18_df[6] #assign a new table name

#since some of the district name is too general, an extra column is added to specify the address for foursquare search
hkdistrict18_data['geo_address']=hkdistrict18_data['District'].str.cat(hkdistrict18_data['Region'],sep= ",")

#drop unwanted columns
hkdistrict18_data1=hkdistrict18_data.drop(['Chinese', 'Area(km2)','Comparable Territory'], axis=1)

hkdistrict18_data1.head


# ### Let's focus on the Hong Kong Island Region

# In[7]:


#identify the rows that applicable to Hong Kong Island Region
hkisland_df=hkdistrict18_data1.loc[hkdistrict18_data1['Region']=='Hong Kong Island']
hkisland_df


# ### Visualize the 4 districts of Hong Kong Island on the Hong Kong map

# In[8]:


from geopy.exc import GeocoderTimedOut 
from geopy.geocoders import Nominatim 
   
# declare an empty list to store 
# latitude and longitude of values  
# of city column 
longitude = [] 
latitude = [] 
   
# function to find the coordinate 
# of a given city  
def findGeocode(address): 
       
    # try and catch is used to overcome 
    # the exception thrown by geolocator 
    # using geocodertimedout   
    try: 
          
        # Specify the user_agent as your 
        # app name it should not be none 
        geolocator = Nominatim(user_agent="foursquare_agent") 
          
        return geolocator.geocode(address) 
      
    except GeocoderTimedOut:
        
         return findGeocode(address)     
  
# each value from city column 
# will be fetched and sent to 
# function find_geocode    
for i in (hkisland_df["geo_address"]): 
      
    if findGeocode(i) != None: 
           
        loc = findGeocode(i) 
          
        # coordinates returned from  
        # function is stored into 
        # two separate list 
        latitude.append(loc.latitude) 
        longitude.append(loc.longitude) 
       
    # if coordinate for a city not 
    # found, insert "NaN" indicating  
    # missing value  
    else: 
        latitude.append(np.nan) 
        longitude.append(np.nan) 


# In[9]:


# Find out the latitude and longitude for the 4 districts of Hong Kong Island Region & create a new table by combining with the exisitng tableisting table
longitude1=pd.DataFrame(longitude)
longitude1.columns=['Longitude']
latitude=pd.DataFrame(latitude)
latitude.columns=['Latitude']


hkisland_df_final= pd.concat([hkisland_df,latitude, longitude1], axis=1)
hkisland_df_final


# #### The coordinate for Southern District is not align with the other 3 districts, for this i revised the coordinate (using Google) and replace the value in the dataframe

# In[10]:


hkisland_df_final.iloc[2, hkisland_df_final.columns.get_loc('Latitude')] = 22.2432
hkisland_df_final.iloc[2, hkisland_df_final.columns.get_loc('Longitude')] = 114.1974
hkisland_df_final


# In[11]:


#visualise the 4 districts of the Hong Kong Island Region

map_hkisland4districts = folium.Map(location=[22.2793278,114.1628131],zoom_start=13)

for lat,lng,District,Region in zip(hkisland_df_final['Latitude'],hkisland_df_final['Longitude'],hkisland_df_final['District'],hkisland_df_final['Region']):
    label = '{}, {}'.format(District, Region)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
    [lat,lng],
    radius=5,
    popup=label,
    color='blue',
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.7,
    parse_html=False).add_to(map_hkisland4districts)
    
map_hkisland4districts


# #### Find out the venue categories and diversity of  each districts

# In[12]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[13]:


CentralNWestern_latitude = hkisland_df_final.loc[0, 'Latitude'] # Chinese & western district latitude value
CentralNWestern_longitude = hkisland_df_final.loc[0, 'Longitude'] # Chinese & western district longitude value

District_name = hkisland_df_final.loc[0, 'District'] # District name

print('Latitude and longitude values of {} are {}, {}.'.format(District_name, 
                                                               CentralNWestern_latitude, 
                                                               CentralNWestern_longitude))


# In[14]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 2000 # define radius

# creatURL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    CentralNWestern_latitude, 
    CentralNWestern_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[15]:


results = requests.get(url).json()
results


# #### While researching online, I found a much simplier python package that can retrieve the foursquare APIs, all you have to do is put in the lat and lng parameter

# In[16]:


get_ipython().system('pip install foursquare')

get_ipython().system('pip install git+https://github.com/dacog/foursquare_api_tools.git#egg=foursquare_api_tools')

import foursquare as fs
from foursquare_api_tools import foursquare_api_tools as ft


# In[17]:


client = fs.Foursquare(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, version=VERSION)


# In[18]:


southern_df=ft.venues_explore(client,lat='22.243200',lng='114.197400',limit=100)

southern_df['District']='Southern'
southern_df


# In[19]:


print('There are {} uniques categories.'.format(len(southern_df['Category'].unique())))


# In[20]:


eastern_df=ft.venues_explore(client,lat='22.283121',lng='114.224180',limit=100)

eastern_df['District']='Eastern'
eastern_df


# In[21]:


wanchai_df=ft.venues_explore(client,lat='22.279015',lng='114.172483',limit=100)
wanchai_df['District']='Wan Chai'
wanchai_df


# In[22]:


central_western_df=ft.venues_explore(client,lat='22.281829',lng='114.158278',limit=100)
central_western_df['District']='Central & Western'
central_western_df


# In[23]:


print('There are {} uniques categories in Southern District.'.format(len(southern_df['Category'].unique())))
print('There are {} uniques categories in Eastern District.'.format(len(eastern_df['Category'].unique())))
print('There are {} uniques categories in Wan Chai District.'.format(len(wanchai_df['Category'].unique())))
print('There are {} uniques categories in Central & Western District.'.format(len(central_western_df['Category'].unique())))


# ### Lets generate some word clouds for the venue categories and see the frequency visually

# In[24]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

stopwords=set(STOPWORDS)


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=1000,
        max_font_size=40, 
        scale=4,
        random_state=0 #chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[25]:


show_wordcloud(southern_df['Category'])


# In[26]:


stopwords.add('dtype')
stopwords.add('Name')
stopwords.add('object')
stopwords.add('Venue')
stopwords.add('Length')
stopwords.add('Category')

show_wordcloud(southern_df['Category'])
show_wordcloud(eastern_df['Category'])
show_wordcloud(central_western_df['Category'])
show_wordcloud(wanchai_df['Category'])


# #### Lets repeat the FourSquare API process and focus on 'Restaurant' this time

# In[27]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 2000 # define radius

categoryId='4bf58dd8d48988d111941735'

HKI_latitude= 22.2588

HKI_longitude=114.1911

# creatURL
HKI_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&categoryId={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    categoryId,
    22.281829,
    114.158278,
    radius, 
    LIMIT)

HKI_url # display URL


# In[28]:


results = requests.get(HKI_url).json()
results


# #### Lets create a new Hong Kong Island Restaurant dataframe by combing the 4 district tables and filter the Column 'Category' with the word 'Restaurant'

# In[29]:


HKI_restaurant_df= pd.concat([southern_df, eastern_df, central_western_df,wanchai_df])

HKI_restaurant_df=HKI_restaurant_df[HKI_restaurant_df['Category'].str.contains("Restaurant")]

HKI_restaurant_df


# #### Lets visualize all these restaurants (from FourSquare API) on the Hong Kong Island Map

# In[30]:


map_HKIrestaurants = folium.Map(location=[22.2793278,114.1628131], zoom_start=12)

# add markers to map
for lat, lng, District, City in zip(HKI_restaurant_df['Latitude'], HKI_restaurant_df['Longitude'], HKI_restaurant_df['District'], HKI_restaurant_df['City']):
    label = '{}, {}'.format(District, City)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_HKIrestaurants)  
    
map_HKIrestaurants


# #### Lets analyze each of the 4 districts

# In[31]:


# one hot encoding
restaurant1_onehot = pd.get_dummies(HKI_restaurant_df[['Category']], prefix="", prefix_sep="")

restaurant1_onehot['District'] = HKI_restaurant_df['District'] 

# move neighborhood column to the first column
fixed_columns = [restaurant1_onehot.columns[-1]] + list(restaurant1_onehot.columns[:-1])
restaurant1_onehot = restaurant1_onehot[fixed_columns]

restaurant1_onehot


# In[32]:


restaurant1_onehot['District'].value_counts().plot(kind='bar')


# In[33]:


restaurant1_grouped = restaurant1_onehot.groupby('District').mean().reset_index()
restaurant1_grouped


# In[34]:


num_top_venues = 5

for hood in restaurant1_grouped['District']:
    print("----"+hood+"----")
    temp = restaurant1_grouped[restaurant1_grouped['District'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Top 5 common venues of the 4 districts

# In[35]:


num_top_venues = 5


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[36]:


num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['District']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
District_venues_sorted = pd.DataFrame(columns=columns)
District_venues_sorted['District'] = restaurant1_grouped['District']

for ind in np.arange(restaurant1_grouped.shape[0]):
    District_venues_sorted.iloc[ind, 1:] = return_most_common_venues(restaurant1_grouped.iloc[ind, :], num_top_venues)

District_venues_sorted.head()


# ### When I try to use K-means clustering, I realised 4 districts are not enough so I have decided to rework the restaurant dataframe and add in a new column 'Neighborhood' so that we can divide our data in a deeper level

# #### I exported the original *HKI_restaurant_df* dataframe and modified the data before importing the csv back to JupyterLab

# In[38]:


HKI_restaurant_df.to_excel(r'C:\Users\ASUS\Downloads\HKI_restaurante.xlsx', index = False)


# In[39]:


HKI_restaurant_revised=pd.read_csv(r'C:\Users\ASUS\Downloads\HKIrestaurantAPI_category.csv')
HKI_restaurant_revised.head()


# In[40]:


HKI_restaurant_revised.groupby('Neighborhood').count()
HKI_restaurant_revised.drop(['District'], axis=1)
HKI_restaurant_revised


# In[41]:


print('There are {} uniques categories.'.format(len(HKI_restaurant_revised['Category'].unique())))


# #### Analyze Each Neighbohood

# In[42]:


# one hot encoding
HKI_restaurant_revised_onehot = pd.get_dummies(HKI_restaurant_revised[['Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
HKI_restaurant_revised_onehot['Neighborhood'] = HKI_restaurant_revised['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [HKI_restaurant_revised_onehot.columns[-1]] + list(HKI_restaurant_revised_onehot.columns[:-1])
HKI_restaurant_revised_onehot = HKI_restaurant_revised_onehot[fixed_columns]

HKI_restaurant_revised_onehot.head()


# In[43]:


HKI_restaurant_revised_onehot.shape


# #### Group rows by neighborhood & by taking the mean of the frequency of occurrence of each category

# In[44]:


HKI_restaurant_revised_grouped = HKI_restaurant_revised_onehot.groupby('Neighborhood').mean().reset_index()
HKI_restaurant_revised_grouped


# In[45]:


HKI_restaurant_revised_onehot_drop=HKI_restaurant_revised_onehot.drop('Neighborhood',axis=1)
HKI_restaurant_revised_onehot_drop


# #### Each neighborhood along with the top 5 most common venues

# In[46]:


num_top_venues = 5

for hood in HKI_restaurant_revised_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = HKI_restaurant_revised_grouped[HKI_restaurant_revised_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Write a function to sort the venues in descending order

# In[47]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# #### Displaying the top 10 restaurant category for each neighborhood

# In[48]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = HKI_restaurant_revised_grouped['Neighborhood']

for ind in np.arange(HKI_restaurant_revised_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(HKI_restaurant_revised_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# ## Cluster Neighhorhoods

# In[49]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(HKI_restaurant_revised_onehot_drop)
    Sum_of_squared_distances.append(km.inertia_)


# In[50]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# #### According to the elbow method, the optimal K for Hong Kong Island Restaurant K means is 7.

# ### Run k-means to cluster the neighborhood into 5 clusters

# In[51]:


# set number of clusters
kclusters = 7

HKI_restaurant_revised_grouped_clustering = HKI_restaurant_revised_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(HKI_restaurant_revised_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[52]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster labels', kmeans.labels_)

HKIsland_merged = HKI_restaurant_revised

# merge HKIsland_grouped with HKI_restaurant_revised to add latitude/longitude for each neighborhood
HKIsland_merged = HKIsland_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

HKIsland_merged.head() # check the last columns!


# In[ ]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[hkI_latitude, hkI_longitude], zoom_start=13)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(HKIsland_merged['Latitude'], HKIsland_merged['Longitude'], HKIsland_merged['Neighborhood'], HKIsland_merged['Cluster labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### Examine each cluster

# In[54]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 0, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# In[55]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 1, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# In[56]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 2, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# In[57]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 3, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# In[58]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 4, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# In[59]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 5, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# In[60]:


HKIsland_merged.loc[HKIsland_merged['Cluster labels'] == 6, HKIsland_merged.columns[[1] + list(range(7, HKIsland_merged.shape[1]))]]


# ### Food service business receipts indices bar chart

# In[86]:


data = {'Years':  ['2017', '2018','2019','2020'],
        'Business receipts indices': ['108.1', '114.6','107.8','76.5']
         
        }

food_service= pd.DataFrame (data, columns = ['Years','Business receipts indices'])

food_service.astype(float)

food_service.dtypes


# In[101]:


import matplotlib as mpl
import matplotlib.pyplot as plt

food_service["Years"]= food_service.Years.astype(int)
food_service["Business receipts indices"]=food_service["Business receipts indices"].astype(float)


# In[104]:


food_service.plot.bar(x="Years", y="Business receipts indices", rot=70,title ="Hong Kong Food Service business receipt indices from 2017 to 2020")


# ### Chinese Restaurant receipt value from 2017 to 2020 line graph

# In[27]:


# List1  
List_CR = [['2017', 105.4], ['2018', 110.4], 
       ['2019', 99.3], ['2019 7-9', 90.5],['2019 10-12', 89.1], ['2020 1-3', 73.5], 
       ['2020 4-6', 72.1], ['2020 7-9', 59.0] ]
    
C_restaurant = pd.DataFrame(List_CR, columns =['Years', 'Receipt value']) 
C_restaurant


# In[28]:


C_restaurant.plot(x='Years', y='Receipt value',rot=70, marker = 'o', title='Hong Kong Restaurant Receipt indices from 2017 to 2020')


# In[ ]:




