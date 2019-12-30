# Pandas' own variable classification

# complete_cols = {dtype('int64'): Index(['id', 'scrape_id', 'host_id', 'accommodates', 'guests_included',
#         'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
#         'maximum_minimum_nights', 'minimum_maximum_nights',
#         'maximum_maximum_nights', 'availability_30', 'availability_60',
#         'availability_90', 'availability_365', 'number_of_reviews',
#         'number_of_reviews_ltm', 'calculated_host_listings_count',
#         'calculated_host_listings_count_entire_homes',
#         'calculated_host_listings_count_private_rooms',
#         'calculated_host_listings_count_shared_rooms'],
#        dtype='object'),
#  dtype('float64'): Index(['thumbnail_url', 'medium_url', 'xl_picture_url', 'host_acceptance_rate',
#         'host_listings_count', 'host_total_listings_count',
#         'neighbourhood_group_cleansed', 'latitude', 'longitude', 'bathrooms',
#         'bedrooms', 'beds', 'square_feet', 'price', 'minimum_nights_avg_ntm',
#         'maximum_nights_avg_ntm', 'review_scores_rating',
#         'review_scores_accuracy', 'review_scores_cleanliness',
#         'review_scores_checkin', 'review_scores_communication',
#         'review_scores_location', 'review_scores_value', 'reviews_per_month'],
#        dtype='object'),
#  dtype('O'): Index(['listing_url', 'last_scraped', 'name', 'summary', 'space',
#         'description', 'experiences_offered', 'neighborhood_overview', 'notes',
#         'transit', 'access', 'interaction', 'house_rules', 'picture_url',
#         'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
#         'host_response_time', 'host_response_rate', 'host_is_superhost',
#         'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
#         'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
#         'street', 'neighbourhood', 'neighbourhood_cleansed', 'city', 'state',
#         'zipcode', 'market', 'smart_location', 'country_code', 'country',
#         'is_location_exact', 'property_type', 'room_type', 'bed_type',
#         'amenities', 'weekly_price', 'monthly_price', 'security_deposit',
#         'cleaning_fee', 'extra_people', 'calendar_updated', 'has_availability',
#         'calendar_last_scraped', 'first_review', 'last_review',
#         'requires_license', 'license', 'jurisdiction_names', 'instant_bookable',
#         'is_business_travel_ready', 'cancellation_policy',
#         'require_guest_profile_picture', 'require_guest_phone_verification'],
#        dtype='object')}

# manual classification

primary_num_cols = [
    "accommodates", "guests_included", "number_of_reviews", "minimum_nights", "maximum_nights",
    "bathrooms", "bedrooms", "beds", "square_feet",
    'review_scores_rating',
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value',
    "host_response_rate", #"calendar_updated" is a bit tricky; handle it later
    "security_deposit", "cleaning_fee", "extra_people"
]
# use for availability:

time_cols = [
    "host_since",
    "first_review", "last_review"
]

binary_cols = [
    "instant_bookable",
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "is_location_exact",
    "require_guest_profile_picture", "require_guest_phone_verification"
]
# tricky: handle later
# "amenities",

# "is_business_travel_ready", "requires_license": f for every one

secondary_num_cols = [
    "number_of_reviews_ltm", "calculated_host_listings_count", 'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm,' 'reviews_per_month',
    "host_location",
]

# combine city and state
cat_cols = [
    "host_neighbourhood", "neighbourhood_cleansed",
        "street", "city", "state",
        "zipcode", "market", "smart_location",
    "host_response_time", "host_verifications", "cancellation_policy",
    "property_type", "room_type", "bed_type"
]

# get race from  "host_name",
# tricky: handle later
# "latitude", "longitude",
# useless: "host_name", (only first name is included)

text_cols = [
    "name", "summary", "space", "description", "neighborhood_overview", "notes", "transit", "access",
    "interaction", "house_rules",  "host_about"
]

# host_url: get img through link
# host_url and host_name: potentially get race, gender, and age
img_cols = [
    "picture_url", "host_thumbnail_url", "host_picture_url"
]

num_pred_cols = [
    'availability_30', 'availability_60', 'availability_90', 'availability_365',
    "price", "weekly_price", "monthly_price"
]

bin_pred_cols = [
    'has_availability',
]

nan_cols = [
    "host_acceptance_rate", 'host_listings_count', 'host_total_listings_count',
]
