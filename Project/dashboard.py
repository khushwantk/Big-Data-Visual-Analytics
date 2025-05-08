# pip install streamlit pandas numpy plotly scikit-learn umap-learn hdbscan networkx statsmodels
# streamlit run run.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title='World University Rankings Dashboard', page_icon='📊', layout='wide')
DATA_PATH = Path('THE World University Rankings 2016-2025.csv')

@st.cache_data
def load_and_clean(path):
    df = pd.read_csv(path)

    # Basic Cleaning
    df['Year'] = df['Year'].astype(int)
    df['Rank'] = df['Rank'].astype(str).str.replace('=', '').astype(float)

    # --- International Students ---
    # Replace stray '%' with closest valid value per university
    valid_intl = df[df['International Students'] != '%'].groupby('Name')['International Students'].first()
    df['International Students'] = df.apply(
        lambda row: valid_intl[row['Name']] if row['International Students'] == '%' and row['Name'] in valid_intl else row['International Students'],
        axis=1
    )
    df['International Students'] = pd.to_numeric(df['International Students'].astype(str).str.replace('%', '').str.strip(), errors='coerce')

    # --- Gender Ratios ---
    def ratio_to_pct(r):
        if pd.isna(r):
            return np.nan
        parts = [float(x) for x in re.split(r'\D+', str(r)) if x]
        return parts[0] / sum(parts) * 100 if len(parts) >= 2 else np.nan

    df['Female %'] = df['Female to Male Ratio'].apply(ratio_to_pct)
    df['Male %'] = 100 - df['Female %']

    # Smart imputation for missing Female%
    female_by_name = df.groupby('Name')['Female %'].mean()
    female_by_country = df.groupby('Country')['Female %'].mean()
    mask = df['Female %'].isna()
    df.loc[mask, 'Female %'] = df.loc[mask, 'Name'].map(female_by_name)
    mask = df['Female %'].isna()
    df.loc[mask, 'Female %'] = df.loc[mask, 'Country'].map(female_by_country)

    # Recalculate Male % after imputation
    df['Male %'] = 100 - df['Female %']

    # Round to nearest integer percentages
    df['Female Ratio'] = df['Female %'].round(0).astype('Int64')
    df['Male Ratio'] = 100 - df['Female Ratio']

    # --- Students to Staff Ratio ---
    df['Students to Staff Ratio'] = pd.to_numeric(df['Students to Staff Ratio'], errors='coerce')
    df.loc[df['Students to Staff Ratio'] > 100, 'Students to Staff Ratio'] = np.nan

    # --- Basic Column Cleanup ---
    df['Country'] = df['Country'].str.strip()

    # Drop raw columns we don't need
    df.drop(columns=['Female to Male Ratio'], inplace=True)

    return df


# CONTINENT MAPPING
asia = ['China', 'Japan', 'Singapore', 'South Korea', 'Turkey',
        'India', 'Iran', 'Malaysia', 'Taiwan', 'Thailand',
        'Pakistan', 'Jordan', 'Kazakhstan', 'Philippines',
        'Vietnam', 'Sri Lanka', 'Hong Kong', 'Brunei Darussalam',
        'Indonesia', 'Bangladesh', 'Russian Federation', 'Iraq',
        'Azerbaijan', 'Israel', 'Saudi Arabia', 'Macao', 'Lebanon',
        'Qatar', 'Oman', 'United Arab Emirates', 'Kuwait', 'Nepal', 'Palestine']

africa = ['South Africa', 'Uganda', 'Egypt', 'Ghana', 'Morocco',
          'Algeria', 'Tunisia', 'Kenya', 'Botswana', 'Ethiopia',
          'Zimbabwe', 'Namibia', 'Tanzania', 'Mozambique',
          'Nigeria', 'Mauritius', 'Zambia']

europe = ['United Kingdom', 'Switzerland', 'Sweden', 'Germany',
          'Belgium', 'Austria', 'Spain', 'Portugal', 'Norway',
          'Bulgaria', 'Ireland', 'Italy', 'Czech Republic',
          'Greece', 'Estonia', 'Cyprus', 'Hungary', 'Slovakia',
          'Ukraine', 'Latvia', 'Lithuania', 'Serbia', 'Montenegro',
          'Kosovo', 'North Macedonia', 'Bosnia and Herzegovina',
          'France', 'Netherlands', 'Finland', 'Denmark', 'Romania',
          'Iceland', 'Luxembourg', 'Poland', 'Slovenia', 'Georgia',
          'Croatia', 'Armenia', 'Malta', 'Belarus', 'Northern Cyprus']

north_america = ['United States', 'Canada', 'Mexico', 'Puerto Rico', 'Jamaica']

oceania = ['Australia', 'New Zealand', 'Fiji']

south_america = ['Brazil', 'Argentina', 'Chile', 'Colombia',
                 'Venezuela', 'Peru', 'Ecuador', 'Uruguay',
                 'Paraguay', 'Bolivia', 'Costa Rica', 'Cuba']

def assign_continent(country):
    if country in asia:
        return 'Asia'
    elif country in africa:
        return 'Africa'
    elif country in europe:
        return 'Europe'
    elif country in north_america:
        return 'North America'
    elif country in oceania:
        return 'Oceania'
    elif country in south_america:
        return 'South America'
    else:
        return 'Unknown'


DF = load_and_clean(DATA_PATH)


DF['Continent'] = DF['Country'].apply(assign_continent)

# COPY DATA FOR FILTERING
# _df = DF.copy()
selected_vars = ['Overall Score', 'Teaching', 'Research Environment', 'Research Quality', 'Industry Impact']

df = DF.copy()

# SIDEBAR FILTERS
st.sidebar.success("✅ Dataset loaded and cleaned!")
# st.balloons()
st.sidebar.header('Filters')
years = ['All'] + sorted(DF['Year'].unique().astype(str))
sel_year = st.sidebar.selectbox('Year', years, index=len(years)-1)
if sel_year != 'All':
    df = df[df['Year'] == int(sel_year)]

sel_ctry = st.sidebar.multiselect('Country', sorted(df['Country'].unique()))
if sel_ctry:
    df = df[df['Country'].isin(sel_ctry)]
rank_rng = st.sidebar.slider('Rank range', int(df.Rank.min()), int(df.Rank.max()), (int(df.Rank.min()), int(df.Rank.max())))
score_rng = st.sidebar.slider('Overall Score range', 0.0, 100.0, (0.0, 100.0))
df = df[df['Rank'].between(*rank_rng) & df['Overall Score'].between(*score_rng)]
st.sidebar.success("✅ Filters successfully applied!!")


st.title("🎓 THE World University Ranking Analysis 2016-2025")


# METRICS
c1, c2, c3, c4 = st.columns(4)
c1.metric('Universities', df.Name.nunique())
c2.metric('Countries', df.Country.nunique())
c3.metric('Median Rank', int(df.Rank.median()))
c4.metric('Mean Score', '{:.1f}'.format(df['Overall Score'].mean()))

# TABS
tabs = st.tabs(['Overview', 'Top Country/Continent', 'Animated World Map', 'Diversity', 'Research & Industry', 'Pairwise', 'Clusters', 'More Insights', 'University Comparer','Conclusions','View Data','EDA'])



## Overview
tab0 = tabs[0]

# Top 20......will change with sidebar-filter
tab0.subheader('Top 20 Universities by Rank ')
# tab0.markdown('Shows the best institutions globally with their scores (Selected country to be country specific)')
top20 = df.nsmallest(20, 'Rank').sort_values('Rank')
fig0 = px.bar(top20, x='Rank', y='Name', orientation='h',
              hover_data=['Overall Score'], title='Top 20 Universities (Selected country to be country specific)')
fig0.update_yaxes(dtick=1,autorange='reversed')
# fig0.update_xaxes(dtick=1)
tab0.plotly_chart(fig0, use_container_width=True)
tab0.markdown('---')

# Bottom 10..........will change with sidebar_filter
tab0.subheader('Bottom 10 Universities by Rank')
# tab0.markdown('Shows the institutions ranked lowest globally with their scores (Selected country to be country specific)')
bottom10 = df.nlargest(10, 'Rank')
fig_bottom10 = px.bar(bottom10, x='Overall Score', y='Name', orientation='h',
                      hover_data=['Rank'], title='Bottom 10 Universities (Selected country to be country specific)')
fig_bottom10.update_yaxes(dtick=1,autorange='reversed')
fig_bottom10.update_xaxes(dtick=1)
tab0.plotly_chart(fig_bottom10, use_container_width=True)
tab0.markdown('---')


# Year wise Trajecory....Won't change with sidebar_filter
tab0.subheader('Rank Trajectories (2016-2025)')
sel_uni = tab0.multiselect('Select universities for trajectory', DF.Name.unique(), default=top20.Name.tolist())
tra = DF[DF.Name.isin(sel_uni)].sort_values(['Name', 'Year'])
fig1 = px.line(tra, x='Year', y='Rank', color='Name', markers=True, title='Rank Trajectories (Selected country to be country specific)')
fig1.update_yaxes(autorange='reversed')
tab0.plotly_chart(fig1, use_container_width=True)
tab0.markdown('---')


df_anim = (
    DF
      .sort_values(['Year','Overall Score'], ascending=[True, False])
      .groupby('Year', group_keys=False)
      .head(20)
      .reset_index(drop=True)
)

# 2) Build the animated bar with a slider only
fig_anim = px.bar(
    df_anim.sort_values(['Year','Overall Score']),
    x='Overall Score',
    y='Name',
    orientation='h',
    animation_frame='Year',
    animation_group='Name',
    range_x=[0,100],
    title='Top 20 Universities by Overall Score Worldwide',
    height=600
)
fig_anim.update_layout(
    yaxis={'categoryorder':'total ascending'},
    updatemenus=[]
)

tab0.plotly_chart(fig_anim, use_container_width=True)



## Country and Continent
tab1 = tabs[1]


year_min, year_max = int(DF.Year.min()), int(DF.Year.max())
year_range = tab1.slider('Select Year Range', year_min, year_max, (year_max-2, year_max-2))
df_country = DF[(DF.Year >= year_range[0]) & (DF.Year <= year_range[1])]

tab1.subheader('Contient wise score for the selected years')
continent_avg = df_country.groupby('Continent')['Overall Score'].mean().reset_index()
continent_avg = continent_avg.sort_values(by='Overall Score', ascending=False)
fig_sun = px.sunburst(continent_avg, path=['Continent'], values='Overall Score',title='Average Score by Continent')
fig_sun.update_traces(insidetextorientation='radial')

TOP_N = 5
continent_avg = (
    df_country
      .groupby('Continent')['Overall Score']
      .mean()
      .reset_index()
)
continent_avg['Country']   = ''
continent_avg['University'] = ''

country_avg = (
    df_country
      .groupby(['Continent','Country'])['Overall Score']
      .mean()
      .reset_index()

)
country_avg['University'] = ''


uni_topn = (
    df_country
      .sort_values('Overall Score', ascending=False)
      .groupby('Country')
      .head(TOP_N)
      .loc[:, ['Continent','Country','Name','Overall Score']]
      .rename(columns={'Name':'University'})
)

sun_df3 = pd.concat([continent_avg, country_avg, uni_topn], ignore_index=True)

fig3 = px.sunburst(
    sun_df3,
    path=['Continent','Country','University'],
    values='Overall Score',
    branchvalues='total',
    title=f'Overall Score Continent → Country → Top {TOP_N} universities'
)
fig3.update_traces(maxdepth=2, insidetextorientation='radial')

col1, col2 = tab1.columns(2)

with col1:
    st.plotly_chart(fig_sun, use_container_width=True)

with col2:
    st.plotly_chart(fig3, use_container_width=True)



fig4 = px.treemap(
    sun_df3,
    path=['Continent','Country','University'],
    values='Overall Score',
    branchvalues='total',
    title=f'Overall Score Continent → Country → Top {TOP_N} universities'
)
fig4.update_traces(maxdepth=2)
tab1.plotly_chart(fig4, use_container_width=True)

# Compare countries Overall Score
cs = df_country.groupby('Country')['Overall Score'].sum().nlargest(10).reset_index()
sel_cmp = tab1.multiselect('**Select countries to compare Overall Score (summed)**', sorted(df_country['Country'].unique()), default=cs['Country'].head(4).tolist())
if sel_cmp:
    cmp_df = df_country[df_country['Country'].isin(sel_cmp)].groupby('Country')['Overall Score'].sum().reindex(sel_cmp).reset_index()
    fig_cmp = px.bar(cmp_df, x='Country', y='Overall Score', title='Comparison of Overall Score for Selected Countries for the selected years')
    tab1.plotly_chart(fig_cmp, use_container_width=True)



# Top 10 by count
ct = df_country.Country.value_counts().nlargest(10).reset_index()
ct.columns = ['Country', 'Count']
fig2 = px.bar(ct, x='Count', y='Country', orientation='h', title='Top 10 Countries by University Count for the selected years')
fig2.update_yaxes(dtick=1,autorange='reversed')
# fig2.update_xaxes(dtick=1)
tab1.plotly_chart(fig2, use_container_width=True)

# Avg Overall Score horizontal bar
cs = df_country.groupby('Country')['Overall Score'].mean().nlargest(10).reset_index()
fig_avg_score = px.bar(cs, x='Overall Score', y='Country', orientation='h', title='Avg Overall Score by Country for the selected years')
fig_avg_score.update_yaxes(dtick=1,autorange='reversed')
tab1.plotly_chart(fig_avg_score, use_container_width=True)

# Avg Overall Score scatter
fig3 = px.scatter(cs, x='Country', y='Overall Score', size='Overall Score', title='Avg Overall Score by Country for the selected years')
# fig3.update_yaxes(dtick=1)
tab1.plotly_chart(fig3, use_container_width=True)




# Top 10 Countries by Industry Impact
ii = df_country.groupby('Country')['Industry Impact'].mean().nlargest(10).reset_index()
fig_ii = px.bar(ii, x='Industry Impact', y='Country', orientation='h', title='Top 10 Countries by Industry Impact for the selected years')
fig_ii.update_yaxes(autorange='reversed')
tab1.plotly_chart(fig_ii, use_container_width=True)

# Top 10 by Students to Staff Ratio
ss = df_country.groupby('Country')['Students to Staff Ratio'].mean().nlargest(10).reset_index()
fig_ss = px.bar(ss, x='Students to Staff Ratio', y='Country', orientation='h', title='Top 10 Countries by Student-Staff Ratio for the selected years')
fig_ss.update_yaxes(autorange='reversed')
tab1.plotly_chart(fig_ss, use_container_width=True)

# Top 10 by Student Population
sp = df_country.groupby('Country')['Student Population'].mean().nlargest(10).reset_index()
fig_sp = px.bar(sp, x='Student Population', y='Country', orientation='h', title='Top 10 Countries by Student Population for the selected years')
fig_sp.update_yaxes(autorange='reversed')
tab1.plotly_chart(fig_sp, use_container_width=True)





## World Map
tab2 = tabs[2]


tab2.subheader('Global Choropleth Maps')

uni_year = (
    DF
      .groupby(['Year','Country'])
      .size()
      .reset_index(name='Universities')
)
map_anim1 = px.choropleth(
    uni_year,
    locations='Country',
    locationmode='country names',
    color='Universities',
    projection='natural earth',
    animation_frame='Year',
    title='Count of Universities Over Time'
)
map_anim1.update_layout(
    geo=dict(showcoastlines=True),
    height=600, width=800
)
tab2.plotly_chart(map_anim1, use_container_width=True)


intl_year = (
    DF
      .groupby(['Year','Country'])['International Students']
      .mean()
      .reset_index()
)
map_anim2 = px.choropleth(
    intl_year,
    locations='Country',
    locationmode='country names',
    color='International Students',
    projection='natural earth',
    animation_frame='Year',
    title='Avg % International Students Over Time'
)
map_anim2.update_layout(
    geo=dict(showcoastlines=True),
    height=600, width=800
)
tab2.plotly_chart(map_anim2, use_container_width=True)


female_year = (
    DF
      .groupby(['Year','Country'])['Female %']
      .mean()
      .reset_index()
)
map_anim3 = px.choropleth(
    female_year,
    locations='Country',
    locationmode='country names',
    color='Female %',
    projection='natural earth',
    animation_frame='Year',
    title='Avg % Female Students Over Time'
)
map_anim3.update_layout(
    geo=dict(showcoastlines=True),
    height=600, width=800
)
tab2.plotly_chart(map_anim3, use_container_width=True)


map_anim1 = px.choropleth(DF, locations='Country', locationmode='country names',
                           color='Student Population', hover_name='Country',
                           projection='natural earth', animation_frame='Year',
                           title='Student Population Over Time')
map_anim1.update_layout(geo=dict(showcoastlines=True), height=600, width=800)
tab2.plotly_chart(map_anim1, use_container_width=True)

    # Animated Overall Score & Industry Impact
map_anim2 = px.choropleth(DF, locations='Country', locationmode='country names',
                            color='Overall Score', hover_name='Country',
                            hover_data={'Industry Impact': True},
                            projection='natural earth', animation_frame='Year',
                            title='Country Distribution Based on Overall Score and Industry Impact')
map_anim2.update_layout(geo=dict(showcoastlines=True), height=600, width=800)
tab2.plotly_chart(map_anim2, use_container_width=True)




## Diversity
tab3 = tabs[3]

tab3.subheader('Mean Gender Diversity Over Time Worldwide')
dfem = DF.groupby('Year')[['Female %', 'Male %']].mean().reset_index()
fig4 = px.line(dfem, x='Year', y=['Female %', 'Male %'], title='Gender Balance Over Time')
fig4.update_yaxes(range=[40, 52])
tab3.plotly_chart(fig4, use_container_width=True)
tab3.markdown('---')


# Top 10 countries by historical Female %
top5_gender = (
    DF
    .groupby('Country')['Female %']
    .mean()
    .nlargest(10)
    .index
    .tolist()
)

# Evolution of Female % in those top 5
dfem_top5 = (
    DF[DF['Country'].isin(top5_gender)]
    .groupby(['Year','Country'])['Female %']
    .mean()
    .reset_index()
)

tab3.subheader('Evolution in Diversity')


fig_top5_gender = px.line(
    dfem_top5,
    x='Year',
    y='Female %',
    color='Country',
    markers=True,
    title='Top Countries by Average Female % : Yearly Trend'
)
tab3.plotly_chart(fig_top5_gender, use_container_width=True)



# Top 10 countries by historical Male %
top5_gender_male = (
    DF
    .groupby('Country')['Male %']
    .mean()
    .nlargest(10)
    .index
    .tolist()
)

# Evolution of Female % in those top 5
dmale_top5 = (
    DF[DF['Country'].isin(top5_gender_male)]
    .groupby(['Year','Country'])['Female %']
    .mean()
    .reset_index()
)

fig_top5_gender = px.line(
    dmale_top5,
    x='Year',
    y='Female %',
    color='Country',
    markers=True,
    title='Top  Countries by Average Male %: Yearly Trend'
)
tab3.plotly_chart(fig_top5_gender, use_container_width=True)

# Evolution of Top 5 Countries Hosting International Students
top_ci = DF.groupby('Country')['International Students'].sum().nlargest(10).index
grouped_ci = DF[DF['Country'].isin(top_ci)].groupby(['Country', 'Year'])['International Students'].sum().reset_index()
fig_topci = px.line(grouped_ci, x='Year', y='International Students', color='Country',
                    markers=True, title='Top Countries Hosting International Students')
tab3.plotly_chart(fig_topci, use_container_width=True)





# Will change by changing left_sidebar
tab3.subheader('Students to Staff Ratio vs Teaching Score for the selected Year and Country')
fig5 = px.scatter(df, x='Students to Staff Ratio', y='Teaching', trendline='ols',
                  title='Staff Ratio vs Teaching',hover_name='Name',)
tab3.plotly_chart(fig5, use_container_width=True)

# Will change by changing left_sidebar
tab3.subheader('Student Population vs Rank for the selected Year and Country')
fig8 = px.scatter(df, x='Student Population', y='Rank', color='Country', size='Overall Score',hover_name='Name')
fig8.update_yaxes(autorange='reversed')
tab3.plotly_chart(fig8, use_container_width=True)



## Research & Industry

tab4 = tabs[4]

# Won't change by changing left_sidebar
tab4.subheader('Average Metrics Over Time Worldwide')
time_df = DF.groupby('Year')[selected_vars].mean().reset_index()
fig17 = px.area(time_df, x='Year', y=selected_vars)
tab4.plotly_chart(fig17, use_container_width=True)

tab4.subheader('Research Quality vs Overall Score for the selected Year and Country')
fig6 = px.scatter(df, x='Research Quality', y='Overall Score', trendline='ols',hover_name='Name')
tab4.plotly_chart(fig6, use_container_width=True)
tab4.markdown('---')
tab4.subheader('Industry Impact vs Research Environment for the selected Year and Country')
fig7 = px.scatter(df, x='Industry Impact', y='Research Environment', size='Overall Score',hover_name='Name',trendline='ols')
tab4.plotly_chart(fig7, use_container_width=True)
tab4.markdown('---')

# Avg Research Quality by country
rq = df_country.groupby('Country')['Research Quality'].mean().nlargest(10).reset_index()
fig_rq = px.bar(rq, x='Research Quality', y='Country', orientation='h', title='Avg Research Quality by Country for the selected Year')
fig_rq.update_yaxes(autorange='reversed')
tab4.plotly_chart(fig_rq, use_container_width=True)


tab4.subheader('Distribution of Core Metrics')
metrics_df = df[selected_vars].melt(var_name='Metric', value_name='Value')
fig9 = px.box(metrics_df, x='Metric', y='Value', title='Core Metrics Distribution wrt selected Year and Country')
tab4.plotly_chart(fig9, use_container_width=True)
tab4.markdown('---')

selm = tab4.multiselect('Choose Metrics', selected_vars, default=['Overall Score', 'Teaching', 'Research Environment', 'Research Quality', 'Industry Impact'])
if selm:
    # melt2 = df[selm].melt(var_name='Metric', value_name='Value')
    melt2 = df[['Name'] + selm].melt(id_vars='Name', var_name='Metric', value_name='Value')
    fig10 = px.violin(melt2, x='Metric', y='Value', box=True, points='all',hover_name='Name')
    tab4.plotly_chart(fig10, use_container_width=True)





## Pairwise
pair_wise = tabs[5]

pair_wise.subheader('Pair Plot of Metrics')
fig11 = px.scatter_matrix(df, dimensions=selected_vars, title='Pair Plot of Metrics for the selected Year and country', height=1400, width=1400,hover_name='Name')
fig11.update_traces(marker=dict(size=3))
fig11.update_layout(font=dict(size=9), margin=dict(l=100, r=100, t=100, b=100), xaxis=dict(nticks=10))
pair_wise.plotly_chart(fig11, use_container_width=True)
pair_wise.markdown('---')

# Correlation Heatmap
numeric_cols = selected_vars + ['Rank', 'Student Population', 'Students to Staff Ratio', 'International Students', 'Female %', 'Male %']
corr = df[numeric_cols].corr()
fig12 = px.imshow(corr, text_auto=True, title='Correlation Heatmap for the selected Year and country', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect='auto')
fig12.update_layout(width=750, height=750, margin=dict(l=100, r=100, t=100, b=100), font=dict(size=12), xaxis=dict(tickangle=45))
pair_wise.plotly_chart(fig12, use_container_width=False)

# Correlation of Overall Score with Factors
columns_of_interest = [
    'Teaching',
    'Research Environment',
    'Research Quality',
    'Industry Impact',
    'International Outlook'
]

corr_vals = (
    df[columns_of_interest + ['Overall Score']]
      .corr()['Overall Score']
      .drop('Overall Score')
      .reindex(columns_of_interest)
)
df_corr = (
    corr_vals
      .reset_index()
      .rename(columns={'index':'Metric', 'Overall Score':'Correlation with Overall Score'})
)

fig_corr_curve = px.line(
    df_corr,
    x='Metric',
    y='Correlation with Overall Score',
    markers=True,
    # line_shape='spline',
    title='Correlation of Overall Score for the selected Year and country'
)

fig_corr_curve.update_xaxes(tickangle=-45)

pair_wise.plotly_chart(fig_corr_curve, use_container_width=True)




## Clusters & PCA
cluster_tab = tabs[6]
cluster_tab.subheader('Clusters and PCA Analysis for the selected Year and country')


cols = cluster_tab.multiselect(
    'Choose Metrics',
    selected_vars,
    default=selected_vars,
    key='cluster_choose_metrics'
)

data_c = df.dropna(subset=cols)
X = StandardScaler().fit_transform(data_c[cols])

inertias = []
for i in range(2, min(11, len(X))):  # only calculate if enough samples
    km = KMeans(n_clusters=i, random_state=0, n_init='auto')
    km.fit(X)
    inertias.append(km.inertia_)

fig_elbow = px.line(
    x=list(range(2, 2 + len(inertias))),
    y=inertias,
    markers=True,
    title='Elbow Method: Inertia vs k '
)
fig_elbow.update_layout(xaxis_title='k', yaxis_title='Inertia')
cluster_tab.plotly_chart(fig_elbow, use_container_width=True)
cluster_tab.markdown('---')

k = cluster_tab.slider(
    'Select number of clusters (k)',
    min_value=2, max_value=10, value=6,
    key='cluster_k'
)
if len(X) < k:
    cluster_tab.error(f"❗ Not enough universities ({len(X)}) to create {k} clusters. Please lower the number of clusters or adjust your filters.")
else:
    cluster_tab.write(f"🔹 **Using k = {k}**")

    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X)
    labels = kmeans.labels_

    import pandas as pd
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    fig_counts = px.bar(
        y=cluster_counts.index.astype(str),
        x=cluster_counts.values,
        labels={'x':'Count', 'y':'Cluster Label'},
        title=f'Cluster Sizes (k={k})',
        orientation='h'
    )
    cluster_tab.plotly_chart(fig_counts, use_container_width=True)

    # 2D PCA scatter with enriched hover data
    pca2 = PCA(n_components=2).fit_transform(X)
    data_c = data_c.assign(PC1=pca2[:, 0], PC2=pca2[:, 1], Cluster=labels.astype(str))

    # Adding useful columns for hover
    data_c['Name'] = df.loc[data_c.index, 'Name']
    data_c['Country'] = df.loc[data_c.index, 'Country']
    data_c['Rank'] = df.loc[data_c.index, 'Rank']
    data_c['Overall Score'] = df.loc[data_c.index, 'Overall Score']
    data_c['Teaching'] = df.loc[data_c.index, 'Teaching']
    data_c['Research Quality'] = df.loc[data_c.index, 'Research Quality']
    data_c['Industry Impact'] = df.loc[data_c.index, 'Industry Impact']
    data_c['International Students'] = df.loc[data_c.index, 'International Students']
    data_c['Student Population'] = df.loc[data_c.index, 'Student Population']
    data_c['Students to Staff Ratio'] = df.loc[data_c.index, 'Students to Staff Ratio']
    data_c['Female %'] = df.loc[data_c.index, 'Female %']

    fig_pca2 = px.scatter(
        data_c,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=f'2D PCA Cluster Assignment (k={k})',
        labels={'Cluster': 'Cluster Label'},
        hover_data={
            'Name': True,
            'Country': True,
            'Rank': True,
            'Overall Score': True,
            'Teaching': True,
            'Research Quality': True,
            'Industry Impact': True,
            'International Students': True,
            'Student Population': True,
            'Students to Staff Ratio': True,
            'Female %': True
        }
    )
    cluster_tab.plotly_chart(fig_pca2, use_container_width=True)

    cluster_tab.markdown('---')


    # Group universities by Cluster and compute mean of each metric
    cluster_profile = data_c.groupby('Cluster')[[
        'Overall Score',
        'Teaching',
        'Research Quality',
        'Industry Impact',
        'International Students',
        'Student Population',
        'Students to Staff Ratio',
        'Female %'
    ]].mean().round(2)

    cluster_tab.subheader('📊 Cluster Profiles: Cluster Centers Metrics')
    cluster_tab.dataframe(cluster_profile, use_container_width=True)


    cluster_tab.subheader('📋 Sample Cluster Interpretation Summary (for 2025 - Worldwide)')

    cluster_tab.markdown("""
    | **Cluster** | **University Type**              | **Key Characteristics** |
    |:------------|:----------------------------------|:-------------------------|
    | **0**       | Emerging Institutions             | Low scores in teaching, research, and international outlook; limited global presence. |
    | **1**       | Research-Driven Universities      | Strong research quality and industry collaborations; moderate teaching focus; growing internationalization. |
    | **2**       | Specialized Research Centers      | Moderate research excellence; weaker teaching and industry engagement; emerging in visibility. |
    | **3**       | Developing Universities           | Weak across all metrics; very low internationalization; needs significant improvement. |
    | **4**       | Balanced Mid-Tier Universities    | Moderate scores in teaching, research, and industry impact; on a steady path toward global competitiveness. |
    | **5**       | World-Class Elite Universities    | Exceptional across all dimensions: teaching, research, industry impact, and international diversity. |
    """, unsafe_allow_html=True)




    pca3 = PCA(n_components=3).fit_transform(X)
    data_c = data_c.assign(PC3=pca3[:,2])

    fig_pca3 = px.scatter_3d(
        data_c,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        title=f'3D PCA Cluster Assignment (k={k})',
        labels={'Cluster':'Cluster Label'},
        hover_data=['Name', 'Country']
    )
    cluster_tab.plotly_chart(fig_pca3, use_container_width=True)



pca_full = PCA().fit(X)
fig_pca_var = px.bar(
    x=[f'PC{i+1}' for i in range(len(pca_full.explained_variance_ratio_))],
    y=pca_full.explained_variance_ratio_,
    title='Full PCA : Explained Variance Ratio'
)
fig_pca_var.update_layout(xaxis_title='Principal Component', yaxis_title='Explained Variance')
cluster_tab.plotly_chart(fig_pca_var, use_container_width=True)


# Create a DataFrame of PCA loadings
loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca_full.explained_variance_ratio_))],
    index=cols  # <-- these are your selected variables
)

cluster_tab.subheader('🔍 PCA Loadings (Variable Contributions)')
cluster_tab.dataframe(loadings.round(3), use_container_width=True)

cluster_tab.markdown("""
PC 1 explains ~80 % of the variance.
    This means that a single latent dimension (a weighted combination of your five metrics) accounts for roughly four-fifths of all the variation across universities. In practice, that first component often represents a “general quality” axis (e.g. overall strength across teaching, research, industry, internationality).
    PC 2 explains ~10 %, and
    PC 3 another ~6 %,
So by the time you include PC 1 + PC 2, you’ve captured about 90 % of the information in the five original metrics. Adding PC 3 pushes you to ~96 %. After that, PC 4 and PC 5 each contribute only a couple of percent (or less).

PC1 captures the overall university excellence (a combination of teaching, research environment, and industry collaboration).

PC2 captures research intensity or specialization (high research quality regardless of other factors).
""")

## More Insights
more_insights = tabs[7]


import umap
import hdbscan
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

more_insights = tabs[7]

more_insights.header('Advanced Clustering & Networks')

more_insights.subheader('🔹 UMAP + HDBSCAN Clustering')


umap_metrics = more_insights.multiselect(
    'Select Metrics for UMAP & HDBSCAN Clustering',
    selected_vars,
    default=selected_vars,
    key='umap_hdbscan_metrics'
)

if umap_metrics:
    data_umap = df.dropna(subset=umap_metrics)
    X_umap = StandardScaler().fit_transform(data_umap[umap_metrics])


    reducer = umap.UMAP(random_state=42)
    X_embedded = reducer.fit_transform(X_umap)


    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True)
    labels = clusterer.fit_predict(X_embedded)


    data_umap['UMAP1'] = X_embedded[:, 0]
    data_umap['UMAP2'] = X_embedded[:, 1]
    data_umap['Cluster'] = labels.astype(str)


    fig_umap = px.scatter(
        data_umap,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_data=['Name', 'Country'],
        title='UMAP + HDBSCAN Clustering of Universities',
        height=600
    )
    more_insights.plotly_chart(fig_umap, use_container_width=True)


    cluster_sizes = data_umap['Cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['Cluster', 'Count']
    fig_cluster_sizes = px.bar(
        cluster_sizes,
        x='Cluster',
        y='Count',
        title='Cluster Sizes from HDBSCAN',
    )
    more_insights.plotly_chart(fig_cluster_sizes, use_container_width=True)

else:
    more_insights.warning('Please select at least one metric to perform UMAP + HDBSCAN clustering.')

more_insights.markdown("---")



# -----------------------------------------
# -----------------------------------------
more_insights.subheader('🔹 University-to-Country Similarity Network (Based on Real Metrics)')


if umap_metrics:

    selected_uni = more_insights.selectbox(
        'Select a University:',
        sorted(data_umap['Name'].unique()),
        key='selected_uni_real'
    )

    selected_country = more_insights.selectbox(
        'Select a Country:',
        sorted(data_umap['Country'].unique()),
        key='selected_country_real'
    )


    sim_threshold = more_insights.slider(
        'Similarity Threshold (%)',
        min_value=70,
        max_value=99,
        value=90,
        help="Higher threshold means stronger similarity required to draw a link."
    )


    data_real = df.dropna(subset=umap_metrics).copy()
    X_real = StandardScaler().fit_transform(data_real[umap_metrics])

    universities = data_real['Name'].tolist()
    countries = data_real['Country'].tolist()


    G = nx.Graph()

    if selected_uni not in universities:
        more_insights.warning(f"{selected_uni} is not available after metric filtering!")
    else:
        idx_main = universities.index(selected_uni)

        # Add main node
        G.add_node(selected_uni, country=countries[idx_main], score=data_real.iloc[idx_main]['Overall Score'])

        for idx, (univ, country) in enumerate(zip(universities, countries)):
            if country == selected_country and univ != selected_uni:
                sim = cosine_similarity(X_real[idx_main].reshape(1, -1), X_real[idx].reshape(1, -1))[0][0] * 100  # in %

                if sim >= sim_threshold:
                    if univ not in G.nodes:
                        G.add_node(univ, country=country, score=data_real.iloc[idx]['Overall Score'])
                    G.add_edge(selected_uni, univ, weight=sim)


        pos = nx.spring_layout(G, seed=42)


        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_color = []
        node_text = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append(G.nodes[node]['score'])  # Overall Score
            node_text.append(f"{node} ({G.nodes[node]['country']})<br>Score: {G.nodes[node]['score']:.2f}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo='text',
            text=[n.split(',')[0] for n in node_text],
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_color,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Overall Score',
                    xanchor='left',
                ),
                line_width=2
            )
        )

        fig_network = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Similarity Network: {selected_uni} vs Universities in {selected_country}',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )
        )

        more_insights.plotly_chart(fig_network, use_container_width=True)

else:
    more_insights.warning('Please select at least one metric to build the similarity network.')



more_insights.markdown('---')

more_insights.subheader('Pair Plot of Key Metrics 2016-2025 ')
fig20 = px.scatter_matrix(DF, dimensions=selected_vars, color='Year', title='Global Pair Plot',height=1400, width=1400,hover_name='Name')
fig20.update_traces(marker=dict(size=3))
fig20.update_layout(font=dict(size=9), margin=dict(l=100, r=100, t=100, b=100), xaxis=dict(nticks=10))
more_insights.plotly_chart(fig20, use_container_width=True)
more_insights.markdown('---')



## Search
search_tab = tabs[8]
search_tab.subheader('University Comparer')
search_tab.markdown('Select two universities to compare them across different metrics.')

# --- Select two universities
universities = sorted(DF['Name'].unique())
selected_unis = search_tab.multiselect(
    'Select two universities:',
    universities,
    default=universities[:2],
    max_selections=2,
    key="compare_universities"
)

if len(selected_unis) == 2:
    u_df = DF[DF['Name'].isin(selected_unis)].sort_values('Year')

    # Latest data for radar plot
    latest_u1 = u_df[u_df['Name'] == selected_unis[0]].iloc[-1]
    latest_u2 = u_df[u_df['Name'] == selected_unis[1]].iloc[-1]

    rad_df = pd.DataFrame({
        'Metric': selected_vars,
        selected_unis[0]: [latest_u1[m] for m in selected_vars],
        selected_unis[1]: [latest_u2[m] for m in selected_vars],
    })

    # --- Radar Chart
    fig_s5 = go.Figure()

    fig_s5.add_trace(go.Scatterpolar(
        r=rad_df[selected_unis[0]],
        theta=rad_df['Metric'],
        fill='toself',
        name=selected_unis[0],
        line=dict(color='#1f77b4')  # Blue
    ))
    fig_s5.add_trace(go.Scatterpolar(
        r=rad_df[selected_unis[1]],
        theta=rad_df['Metric'],
        fill='toself',
        name=selected_unis[1],
        line=dict(color='#ff7f0e')  # Orange
    ))

    fig_s5.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title=f"Profile Radar Comparison ({latest_u1.Year} latest)"
    )
    search_tab.plotly_chart(fig_s5, use_container_width=True)


    color_map = {
        selected_unis[0]: '#1f77b4',  # Blue
        selected_unis[1]: '#ff7f0e',  # Orange
    }

    # --- Rank Over Time
    fig_s1 = px.line(
        u_df,
        x='Year',
        y='Rank',
        color='Name',
        markers=True,
        title='Rank Over Time',
        color_discrete_map=color_map
    )
    fig_s1.update_yaxes(autorange='reversed')
    search_tab.plotly_chart(fig_s1, use_container_width=True)

    # --- Overall Score Over Time
    fig_s2 = px.bar(
        u_df,
        x='Year',
        y='Overall Score',
        color='Name',
        barmode='group',
        title='Overall Score Over Time',
        range_y=[0, 100],
        color_discrete_map=color_map
    )
    search_tab.plotly_chart(fig_s2, use_container_width=True)

    # --- Student Population Over Time
    fig_s3 = px.line(
        u_df,
        x='Year',
        y='Student Population',
        color='Name',
        markers=True,
        title='Student Population Over Time',
        color_discrete_map=color_map
    )
    search_tab.plotly_chart(fig_s3, use_container_width=True)

    # --- Diversity Trends
    fig_s4 = px.line(
        u_df,
        x='Year',
        y=['Female %', 'International Students'],
        color='Name',
        markers=True,
        title='Diversity Trends (Female %, International Students)',
        color_discrete_map=color_map
    )
    search_tab.plotly_chart(fig_s4, use_container_width=True)

    # --- Scatter Matrix
    fig_s6 = px.scatter_matrix(
        u_df,
        dimensions=selected_vars,
        color='Name',
        title='Pair Plot for Selected Metrics',
        height=900,
        width=900,
        color_discrete_map=color_map
    )
    search_tab.plotly_chart(fig_s6, use_container_width=True)


else:
    search_tab.info('Please select exactly two universities to compare.')


## Conclusions
final_story = tabs[9]
final_story.header('Conclusions & Story')
final_story.markdown('''


Over a decade, higher education has undergone substantial transformations. Our visual analytics journey reveals that while the United States and Europe maintain dominance, Asian institutions are significantly improving, particularly in research quality and international appeal.


We observe progressive gender diversity improvements, with female student participation rising consistently, demonstrating effective global policies for inclusivity. However, certain regions still lag behind, emphasizing the need for targeted diversity strategies.

Resource allocation emerges as a critical performance driver. Institutions with lower student-to-staff ratios consistently demonstrate superior teaching quality, urging policymakers to reconsider faculty investments strategically.

Through PCA Loadings, we discover that PC1 acts as a 'University Strength' axis blending teaching, research, and industry outreach, while PC2 distinguishes universities highly focused on research impact. This two-axis structure explains over 90% of the variance, validating the use of PCA scatter plots for meaningful clustering and interpretation of global university profiles.

Techniques like Principal Component Analysis (PCA) and KMeans helped us capture broader patterns and clusters among institutions worldwide. To uncover deeper, non-linear relationships, we applied UMAP and HDBSCAN, which allowed us to isolate hidden clusters and detect outliers with greater accuracy. By constructing similarity networks based on true cosine similarities, we also introduced a new perspective: identifying academic ``twins'' across different countries and continents, something traditional ranking lists cannot easily reveal.

Our clustering analysis notably reveals six distinct university profiles:

Cluster 5 represents globally elite institutions excelling across all metrics.
Cluster 1 universities demonstrate powerful research and industry engagement but moderate teaching.Meanwhile, clusters 0, 2, and 3 consist of emerging universities striving for global prominence but facing challenges in teaching, research, and internationalization.


| **Cluster** | **University Type**              | **Key Characteristics** |
|:------------|:----------------------------------|:-------------------------|
| **0**       | Emerging Institutions             | Low scores in teaching, research, and international outlook; limited global presence. |
| **1**       | Research-Driven Universities      | Strong research quality and industry collaborations; moderate teaching focus; growing internationalization. |
| **2**       | Specialized Research Centers      | Moderate research excellence; weaker teaching and industry engagement; emerging in visibility. |
| **3**       | Developing Universities           | Weak across all metrics; very low internationalization; needs significant improvement. |
| **4**       | Balanced Mid-Tier Universities    | Moderate scores in teaching, research, and industry impact; on a steady path toward global competitiveness. |
| **5**       | World-Class Elite Universities    | Exceptional across all dimensions: teaching, research, industry impact, and international diversity. |


Interactive comparisons between institutions illuminate competitive dynamics, crucial for stakeholders aiming at strategic improvements or partnerships.

Overall, this project goes beyond simply reporting which universities rank highest. It sheds light on how and why certain patterns emerge, evolve, and sometimes diverge over time. The final system offers an educational intelligence framework that is not only informative but also actionable --- serving students, researchers, policymakers, and institutional leaders aiming to better understand and shape the future of higher education.

''')



## View Data
tab_view = tabs[10]
tab_view.subheader("Dataset Explorer")


query = tab_view.text_input("🔍 Search University Name (substring match)", "")


years = sorted(DF['Year'].unique())
sel_years = tab_view.multiselect("Filter Years", years, default=years)
countries = sorted(DF['Country'].unique())
sel_countries = tab_view.multiselect("Filter Countries", countries, default=countries)


all_cols = DF.columns.tolist()
sel_cols = tab_view.multiselect("Select Columns to Display", all_cols, default=all_cols)


df_view = DF[
    (DF['Year'].isin(sel_years)) &
    (DF['Country'].isin(sel_countries))
]
if query:
    df_view = df_view[df_view['Name']
        .str.contains(query, case=False, na=False)]


tab_view.dataframe(df_view[sel_cols], use_container_width=True)



import seaborn as sns
import matplotlib.pyplot as plt

with tabs[11]:
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Columns and Description of original data")
    columns_info = {
        "Rank": "The global ranking of institutions",
        "Name": " The gname of institutions",
        "Country": "The country of each university",
        "Student Population": "Total number of students enrolled",
        "Student-to-Staff Ratio": "Faculty availability for students",
        "International Students": "Percentage of international students",
        "Female-to-Male Ratio": "Gender distribution of students",
        "Overall Score": "Composite ranking score",
        "Teaching Score": "Rating based on teaching quality",
        "Research Environment Score": "Research facilities & output",
        "Research Quality Score": "Research impact in citations",
        "Industry Impact Score": "University collaboration with industries",
        "International Outlook Score": "Diversity in faculty & students",
        "Year (2016-2025)": "Year-wise rankings"
    }
    description_df = pd.DataFrame(list(columns_info.items()), columns=["Attribute", "Description"])
    st.dataframe(description_df)

    st.subheader("Columns Present in Cleaned Data (DF)")
    st.write(f"Total Columns: {len(DF.columns)}")
    columns_list = ", ".join(DF.columns)
    st.write(f" {columns_list}")


    st.subheader("Missing Values Summary")

    missing = DF.isna().sum()
    missing_percent = (missing / len(DF)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_percent
    })


    missing_df = missing_df[missing_df['Missing Count'] > 0]
    missing_df = missing_df.sort_values('Missing Count', ascending=False)

    st.dataframe(missing_df)


    st.subheader("Duplicate Rows")
    st.write(f"Total Duplicates: {DF.duplicated().sum()}")

    st.subheader("Descriptive Statistics")
    st.dataframe(DF.describe())

    st.subheader("Number of Unique Universities")
    unique_universities = DF['Name'].nunique()
    st.write(f"Total Unique Universities: {unique_universities}")

    st.subheader("Top 5 Universities by Overall Score")
    top5_score = DF.sort_values('Overall Score', ascending=False).head(5)
    st.dataframe(top5_score[['Name', 'Country', 'Overall Score','Year']])

    st.subheader("Bottom 5 Universities by Overall Score")
    bottom5_score = DF.sort_values('Overall Score').head(5)
    st.dataframe(bottom5_score[['Name', 'Country', 'Overall Score','Year']])

    st.subheader("Country with Highest and Lowest Average Overall Score")
    country_avg_score = DF.groupby('Country')['Overall Score'].mean().sort_values(ascending=False).reset_index()
    st.metric(label="Top Country by Mean Overall Score", value=country_avg_score.iloc[0]['Country'], delta=f"{country_avg_score.iloc[0]['Overall Score']:.2f}")
    st.metric(label="Last Country by Mean Overall Score", value=country_avg_score.iloc[-1]['Country'], delta=f"{country_avg_score.iloc[-1]['Overall Score']:.2f}")

    st.subheader("Highest and Lowest Overall Score Achieved")
    max_score_row = DF.loc[DF['Overall Score'].idxmax()]
    min_score_row = DF.loc[DF['Overall Score'].idxmin()]
    st.metric(label="Highest Overall Score Achieved", value=max_score_row['Name'], delta=f"{max_score_row['Overall Score']:.2f}")
    st.metric(label="Lowest Overall Score Achieved", value=min_score_row['Name'], delta=f"{min_score_row['Overall Score']:.2f}")


    st.subheader("Distribution of Student Population")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(DF['Student Population'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Students to Staff Ratio Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(DF['Students to Staff Ratio'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Overall Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(DF['Overall Score'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("International Students Percentage Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(DF['International Students'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)


    st.subheader("Female Ratio vs Overall Score")
    fig = px.scatter(DF, x='Female %', y='Overall Score', color='Country', hover_name='Name', width=600, height=400)
    st.plotly_chart(fig, use_container_width=True)


st.caption('Data source: Times Higher Education 2016-2025')
