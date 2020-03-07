# 'steam'
# Steam Video Game and Bundle Data: These datasets contain reviews from the Steam video game platform, 
# and information about which games were bundled together.

# 'google_local'
# Google Local Reviews: These datasets contain reviews about businesses from Google Local (Google Maps). 
# Data includes geographic information for each business as well as reviews.

if [ $1 == 'steam' ]
then
    # Version 1: Review Data (6.7mb)
    # wget http://deepx.ucsd.edu/public/jmcauley/steam/australian_user_reviews.json.gz
    # Version 1: User and Item Data (71mb)
    # wget http://deepx.ucsd.edu/public/jmcauley/steam/australian_users_items.json.gz
    # Version 2: Review Data (1.3gb)
    wget http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
    # Version 2: Item metadata (2.7mb)
    wget http://cseweb.ucsd.edu/~wckang/steam_games.json.gz
    # Bundle Data (92kb)
    wget http://deepx.ucsd.edu/public/jmcauley/steam/bundle_data.json.gz
elif [ $1 == 'google_local' ]
then
    # Places Data (276mb)
    wget http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/places.clean.json.gz
    # User Data (178mb)
    wget http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/users.clean.json.gz
    # Review Data (1.4gb)
    wget http://deepyeti.ucsd.edu/jmcauley/datasets/googlelocal/reviews.clean.json.gz
fi