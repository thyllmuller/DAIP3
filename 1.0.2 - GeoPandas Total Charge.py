import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import D_C_T

pd.set_option('display.max_columns', None)

'''LOADING ALL DATA AND MERGING'''
# my data import
cb = D_C_T.c_Bload()
# convert the abbreviation to full name & reverse the stupid mapping
us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
    "Washington, DC": "DC",
}
# i almost died making that
# reverse the dict and replace the states
inv_dict = {v: k for k, v in us_state_to_abbrev.items()}
cb_rev = cb.replace({"State": inv_dict})
cb_ren = cb_rev.rename(columns={"State": "NAME"})
# load states data
states = geopandas.read_file("data/usa-states-census-2014.shp")
states = states.to_crs("EPSG:3395")

''' MERGING DATA: BE REALLY FUCKING CAREFUL'''
# apply functions first
cb_ren_final = cb_ren.groupby(["NAME"])["Account_Length"].sum().reset_index(name='c_count')
print(cb_ren_final)

# merge our data
geo_merge = states.merge(cb_ren_final, on="NAME")

'''THE PLOT'''
fig = plt.figure(1, figsize=(25, 15))
ax = fig.add_subplot()
geo_merge.apply(
    lambda x: ax.annotate(
        text=("%.0f" % x.c_count),
        xy=x.geometry.centroid.coords[0],
        ha='center', fontsize=10),
    axis=1)
geo_merge.boundary.plot(ax=ax, color='Black', linewidth=.4)
geo_merge.plot(ax=ax, cmap='OrRd', column=geo_merge['c_count'], figsize=(12, 12))
ax.text(-0.05, 0.5, 'label here', transform=ax.transAxes,
        fontsize=20, color='gray', alpha=0.5,
        ha='center', va='center', rotation='90')
ax.margins(0)
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
plt.show()
fig.tight_layout()
fig.savefig('Total_Charge.png')

''' overlay shading:
west = states[states['region'] == 'West']
us_boundary_map = states.boundary.plot(figsize=(18, 12), color="Gray")
west.plot(ax=us_boundary_map,  color="DarkGray")
'''

'''2x annotations.
states.apply(
    lambda x: ax.annotate(
        text=x.NAME + "\n"
             + str(math.floor(x.ALAND / 2589988.1103))
             + " sq mi",
        xy=x.geometry.centroid.coords[0],
        ha='center', fontsize=10), axis=1)
'''
