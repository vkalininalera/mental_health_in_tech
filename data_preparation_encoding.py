import pandas as pd
from fixing_missing_values import data, col_if_yes_diagnosed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import nltk
from collections import Counter
import matplotlib.pyplot as plt


# Ensure NLTK stopwords are available
nltk.download('stopwords')

# Add columns that represent a mental disorder
# 1: Clean and split the diagnosis string into lists
data['diagnosis_list'] = data[col_if_yes_diagnosed].fillna('').apply(
    lambda x: [i.strip() for i in x.split('|')] if x else []
)

# 2: Binarize
mlb = MultiLabelBinarizer()
diag_encoded_df = pd.DataFrame(
    mlb.fit_transform(data['diagnosis_list']),
    columns=mlb.classes_,
    index=data.index
)

# Flatten the list of diagnoses and count frequency
all_diagnoses = [d for sublist in data['diagnosis_list'] for d in sublist]
diagnosis_counts = Counter(all_diagnoses)

# Convert to lists for plotting
labels, values = zip(*diagnosis_counts.most_common())


# Plot 'Most common mental health issues'.
#
plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel("Frequency")
plt.title("Most Common Mental Health Issues")
plt.tight_layout()
plt.show()


# 3: Add the new diagnosis columns
data = pd.concat([data, diag_encoded_df], axis=1)

# 4: Drop helper columns
data.drop(columns=['diagnosis_list', col_if_yes_diagnosed], inplace=True)

# ----- Define TF-IDF text processing function -----
custom_stop_words = {
    'wa', 'wouldnt', 'dont', 'im', 'id', 'like', 'make', 'think', 'want', 'way', 'getting',
    'its', 'ours', 'most', 'youve', 'couldnt', 'would', 'could', 'know', 'may', 'might', 'need',
    'issues', 'see', 'whether', 'employers', 'likely', 'unless', 'bring', 'mental', 'physical',
    'health', 'even', 'though', 'interview', 'much', 'still', 'employer'
}

all_stop_words = set(nltk.corpus.stopwords.words('english')).union(custom_stop_words)

max_words = 100

def process_and_align_text(column, prefix, original_data):
    # Clean text: remove "no response" and "i dont know"
    filtered = column[~column.isin(["no response", "i dont know"])]
    filtered = filtered[filtered.str.split().str.len().le(max_words)].reset_index(drop=True)

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=False,
        stop_words=list(all_stop_words),
        token_pattern=r'(?u)\b\w[\w-]+\b',
        # max_df=0.95,
        # min_df=0.005,
        ngram_range=(3, 4)
    )
    tfidf = vectorizer.fit_transform(filtered)
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f"{prefix}{feat}" for feat in vectorizer.get_feature_names_out()])

    # Align TF-IDF with original data
    aligned_df = pd.DataFrame(0.0, index=original_data.index, columns=tfidf_df.columns)
    aligned_df.iloc[:len(filtered), :] = tfidf_df.values
    return aligned_df


# ----- Process and integrate text columns -----
# Mental health
mental_col = 'Why or why not bring up with a potential employer in an interview(mental health issue)_cleaned?'
tfidf_mental = process_and_align_text(data[mental_col], 'MH_TFIDF_', data)

# 1: Get the mean TF-IDF score per feature
tfidf_means = tfidf_mental.mean().sort_values(ascending=False)

# 2: Plot the top features in mental health comments
top_n = 20
plt.figure(figsize=(10, 6))
tfidf_means.head(top_n).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top TF-IDF Features in Mental Health Comments')
plt.xlabel('Mean TF-IDF Score')
plt.tight_layout()
plt.show()

data = data.drop(columns=[mental_col])


# Physical health
physical_col = 'Why or why not bring up with a potential employer in an interview(physical health issue)_cleaned?'
tfidf_physical = process_and_align_text(data[physical_col], 'PH_TFIDF_', data)

tfidf_means = tfidf_physical.mean().sort_values(ascending=False)
# Plot the top features in physical health comments
top_n = 20
plt.figure(figsize=(10, 6))
tfidf_means.head(top_n).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top TF-IDF Features in Physical Health Comments')
plt.xlabel('Mean TF-IDF Score')
plt.tight_layout()
plt.show()


data = data.drop(columns=[physical_col])

# Replace "Maybe" with "I don't know" only in object columns
data.loc[:, data.select_dtypes(include=['object']).columns] = data.select_dtypes(include=['object']).replace("Maybe",
                                                                                                             "I don't know")
data['Do you feel that being identified as a person with a mental health issue would hurt your career?'] = data[
    ('Do you feel that being identified as a person with a mental health issue would hurt your career?')].replace({
    "No, I don't think it would": "No",
    'No, it has not': "No",
    'Yes, I think it would': "Yes",
    'Yes, it has': "Yes"
})

data[
    ('Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental '
     'health issue?')] = \
    data[
        ('Do you think that team members/co-workers would view you more negatively if they knew you suffered from a '
         'mental health issue?')].replace(
        {
            "No, I don't think they would": "No",
            'Yes, I think they would': "Yes",
            'Yes, they do': "Yes",
            'No, they do not': "No"
        })
data[
    ('Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your '
     'current or previous workplace?')] = \
    data[
        ('Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your '
         'current or previous workplace?')].replace(
        {
            "Maybe/Not sure": "A",
            "Yes, I experienced": "Yes",
            "Yes, I observed": "Yes",
            "No response":"A"
        })
data[
    ('If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:')]=\
data [
     ('If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:')].replace(
        {
            "Very easy": "Easy",
            "Somewhat easy": "Easy",
            "Somewhat difficult":"Difficult",
            "Very difficult": "Difficult",
            "Neither easy nor difficult":"A",
            "I don't know":"A",
            "Not applicable":"A"
        }
     )

data['How willing would you be to share with friends and family that you have a mental illness?'] = data[
    'How willing would you be to share with friends and family that you have a mental illness?'].replace({
    'Very open': 'Open', 'Somewhat open': 'Open', 'Neutral': 'NA', 'Somewhat not open': 'Not open',
    'Not open at all': 'Not open', 'Not applicable to me (I do not have a mental illness)': 'NA'
})

data[
    ('Have your previous employers provided mental health benefits?')]=\
data [
     ('Have your previous employers provided mental health benefits?')].replace(
        {
            "No, none did": "No",
            "Yes, they all did": "Yes",
            "Some did":"Yes",
            "I don't know": "NA",
            "No previous employer":"NA"
        }
     )

data[
    ('Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?')]=\
data [
     ('Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?')].replace(
        {
            "None did": "No",
            "Some did": "Yes",
            "I don't know": "NA",
            "No previous employer":"NA",
            "Yes, they all did":"Yes"
        }
     )


data[
    ('Do you think that discussing a mental health disorder with your employer would have negative consequences?')]=\
data [
     ('Do you think that discussing a mental health disorder with your employer would have negative consequences?')].replace(
        {   "Some of them":"Yes",
            "None of them":"No",
            "I don't know":"A",
            "Yes, all of them":"Yes",
            "No previous employer":"A",
            "Not applicable":"A"
        }
     )


data[
    ('Do you think that discussing a mental health disorder with previous employers would have negative consequences?')]=\
data [
     ('Do you think that discussing a mental health disorder with previous employers would have negative consequences?')].replace(
        {
            "Some of them":"Yes",
            "None of them":"No",
            "I don't know":"A",
            "Yes, all of them":"Yes",
            "No previous employer":"A"
        }
     )



# Map "Yes" to 1 and "No" to 0
data['Have you been diagnosed with a mental health condition by a medical professional?'] = data[
    'Have you been diagnosed with a mental health condition by a medical professional?'].map({'Yes': 1, 'No': 0})

country_mapping = {'United Kingdom': 'Europe', 'Germany': 'Europe', 'Netherlands': 'Europe',
                   'Czech Republic': 'Europe', 'Lithuania': 'Europe', 'France': 'Europe',
                   'Poland': 'Europe', 'Belgium': 'Europe', 'Denmark': 'Europe', 'Sweden': 'Europe',
                   'Russia': 'Europe', 'Spain': 'Europe', 'Norway': 'Europe', 'Ireland': 'Europe',
                   'Italy': 'Europe', 'Finland': 'Europe', 'Slovakia': 'Europe', 'Austria': 'Europe',
                   'Greece': 'Europe', 'Romania': 'Europe', 'Hungary': 'Europe', 'Estonia': 'Europe',
                   'Bosnia and Herzegovina': 'Europe', 'Bulgaria': 'Europe', 'Serbia': 'Europe',
                   'United States of America': 'North America', 'Canada': 'North America',
                   'Mexico': 'North America', 'Costa Rica': 'North America', 'Guatemala': 'North America',
                   'India': 'Asia', 'Vietnam': 'Asia', 'Pakistan': 'Asia', 'Afghanistan': 'Asia',
                   'Iran': 'Asia', 'Israel': 'Asia', 'Taiwan': 'Asia', 'Japan': 'Asia', 'Bangladesh': 'Asia',
                   'Brunei': 'Asia', 'China': 'Asia',
                   'Brazil': 'South America', 'Venezuela': 'South America', 'Argentina': 'South America',
                   'Colombia': 'South America', 'Chile': 'South America', 'Ecuador': 'South America',
                   'Australia': 'Oceania', 'New Zealand': 'Oceania',
                   'South Africa': 'Africa', 'Algeria': 'Africa', 'United Arab Emirates': 'Asia',
                   'Switzerland': 'Europe', 'Turkey': 'Asia'}

data['What country do you live in?'] = data['What country do you live in?'].replace(country_mapping)
data['What country do you work in?'] = data['What country do you work in?'].replace(country_mapping)

data[
    ('Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment '
     'resources with previous employers?')]=\
data [
     ('Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment '
     'resources with previous employers?')].replace(
        {
            "Yes, always":"Yes",
            "Sometimes":"Yes",
            "I don't know":"A",
            "No":"No",
            "No previous employer":"A"
        }
     )

data[
    ('Did you feel that your previous employers took mental health as seriously as physical health?')]=\
data [
     ('Did you feel that your previous employers took mental health as seriously as physical health?')].replace(
        {
            "I don't know":"A",
            "Some did":"Yes",
            "None did":"No",
            "Yes, they all did":"Yes",
            "No previous employer":"A"
        }
     )



data[
    ('If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?')]=\
data [
     ('If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?')].replace(
        {
            "Not applicable to me":"A",
            "Rarely":"Yes",
            "Sometimes":"Yes",
            "Never":"No",
            "Often":"Yes"
        }
     )

data[
    ('If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?')]=\
data [
     ('If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?')].replace(
        {
            "Not applicable to me":"A",
            "Rarely":"Yes",
            "Sometimes":"Yes",
            "Never":"No",
            "Often":"Yes"
        }
     )

data[
    ('Do you feel that your employer takes mental health as seriously as physical health?')]=\
data [
     ('Do you feel that your employer takes mental health as seriously as physical health?')].replace(
        {
            "No":"No",
            "Yes":"Yes",
            "Not applicable":"A",
            "I don't know":"A"
        }
     )

data[
    ('Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment '
     'resources provided by your employer?')]=\
data [
     ('Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment '
     'resources provided by your employer?')].replace(
        {
            "No":"No",
            "Yes":"Yes",
            "Not applicable":"A",
            "I don't know":"A"
        }
     )

data[
    ('Does your employer offer resources to learn more about mental health concerns and options for seeking help?')]=\
data [
     ('Does your employer offer resources to learn more about mental health concerns and options for seeking help?')].replace(
        {
            "No":"No",
            "Yes":"Yes",
            "Not applicable":"A",
            "I don't know":"A"
        }
     )
data[
    ('Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other '
     'official communication)?')]=\
data [
     ('Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other '
     'official communication)?')].replace(
        {
            "No":"No",
            "Yes":"Yes",
            "Not applicable":"A",
            "I don't know":"A"
        }
     )

data[
    ('Did your previous employers provide resources to learn more about mental health issues and how to seek help?')]=\
data [
     ('Did your previous employers provide resources to learn more about mental health issues and how to seek help?')].replace(
        {
            "Non did":"No",
            "Some did":"Yes",
            "No previous employer":"A",
            "Yes, they all did":"Yes"
        }
     )

data[
    ('Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?')]=\
data [
     ('Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?')].replace(
        {
            "No response":"A",
            "Yes":"Yes",
            "No":"No",
            "I don't know":"A"
        }
     )

data[
    ('Were you aware of the options for mental health care provided by your previous employers?')]=\
data [
     ('Were you aware of the options for mental health care provided by your previous employers?')].replace(
        {
            "Not Aware":"No",
            "Aware":"Yes",
            "No History":"A",
        }
     )

data[
    ('Do you think that discussing a physical health issue with previous employers would have negative consequences?')]=\
data [
     ('Do you think that discussing a physical health issue with previous employers would have negative consequences?')].replace(
        {
            "None of them":"No",
            "Some of them":"Yes",
            "Yes, all of them":"Yes",
            "No previous employer":"A"
        }
     )

data[
    ('Would you feel comfortable discussing a mental health disorder with your coworkers?')]=\
data [
     ('Would you feel comfortable discussing a mental health disorder with your coworkers?')].replace(
        {
            "No":"No",
            "Not applicable":"A",
            "I don't know":"A",
            "Yes":"Yes"
        }
     )



data[
    ('Do you think that discussing a physical health issue with your employer would have negative consequences?')]=\
data [
     ('Do you think that discussing a physical health issue with your employer would have negative consequences?')].replace(
        {
            "No":"No",
            "Not applicable":"A",
            "I don't know":"A",
            "Yes":"Yes"
        }
     )



data[
    ('Did your previous employers provide resources to learn more about mental health issues and how to seek help?')]=\
data [
     ('Did your previous employers provide resources to learn more about mental health issues and how to seek help?')].replace(
        {
            "None did":"No",
            "Some did":"Yes",
            "No previous employer":"A",
            "Yes, all of them":"Yes"
        }
     )

data[
    ('Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?')]=\
data [
     ('Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?')].replace(
        {
            "None of them":"No",
            "Some of them":"Yes",
            "No previous workplace":"A",
            "Yes, all of them":"Yes"
        }
     )


data[
    ("Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?")]=\
data [
     ("Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?")].replace(
        {
            "Yes":"Yes",
            "I don't know":"A",
            "Not applicable":"A",
            "No":"No"
        }
     )


columns_drop_first_true = \
    ['Does your employer provide mental health benefits as part of healthcare coverage?',
     'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other '
     'official communication)?',
     'Does your employer offer resources to learn more about mental health concerns and options for seeking help?',
     'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment '
     'resources provided by your employer?',
     'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:',
     'Do you feel that your employer takes mental health as seriously as physical health?',
     'Do you think that discussing a mental health disorder with your employer would have negative consequences?',
     'Do you think that discussing a physical health issue with your employer would have negative consequences?',
     'Would you feel comfortable discussing a mental health disorder with your coworkers?',
     'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
     'Have you heard of or observed negative consequences for co-workers who have been open about mental health '
     'issues in your workplace?', 'Have your previous employers provided mental health benefits?',
     'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other '
     'official communication)?',
     'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment '
     'resources with previous employers?',
     'Did you feel that your previous employers took mental health as seriously as physical health?',
     'Would you be willing to bring up a physical health issue with a potential employer in an interview?',
     'Would you bring up a mental health issue with a potential employer in an interview?',
     'Do you feel that being identified as a person with a mental health issue would hurt your career?',
     'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?',
     'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?',
     'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?',
     'Do you have a family history of mental illness?',
     'Have you had a mental health disorder in the past?',
     'Do you currently have a mental health disorder?',
     'How willing would you be to share with friends and family that you have a mental illness?',
     'Do you think that discussing a mental health disorder with previous employers would have negative consequences?',
     'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?',
     'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?',
     'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?',
     'Did your previous employers provide resources to learn more about mental health issues and how to seek help?',
     'Do you think that discussing a physical health issue with previous employers would have negative consequences?',
     'Were you aware of the options for mental health care provided by your previous employers?'

     ]

columns_drop_first_false = [
                            'What is your gender?', 'What country do you live in?', 'What country do you work in?',
                            'Do you work remotely?', 'How many employees does your company or organization have?']


# Combined function to handle both encoding scenarios
def one_hot_encode_columns(df, columns_drop_first_true, columns_drop_first_false):
    """
    One-Hot Encodes specified columns in the DataFrame with individual drop_first settings.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_drop_first_true (list): Columns where drop_first=True.
    columns_drop_first_false (list): Columns where drop_first=False.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns one-hot encoded as integers.
    """
    encoded_df = df.copy()

    for col in columns_drop_first_true + columns_drop_first_false:
        # Check if the column exists in the DataFrame
        if col not in encoded_df.columns:
            continue

        # Determine if drop_first should be True or False
        drop_first = col in columns_drop_first_true

        # Dynamically find the unique values to order categories correctly
        unique_values = sorted(encoded_df[col].dropna().unique())
        if 'Not applicable' in unique_values:
            unique_values = ['Not applicable'] + [v for v in unique_values if v != 'Not applicable']

        # Set the order of categories dynamically
        encoded_df[col] = pd.Categorical(encoded_df[col], categories=unique_values, ordered=True)

        # Generate One-Hot Encoded DataFrame with integer type (0 or 1)
        prefix = col[:150].replace(' ', '_')  # Shortened prefix to avoid long column names
        dummies = pd.get_dummies(encoded_df[col], prefix=prefix, drop_first=drop_first).astype(int)

        # Concatenate the encoded columns and drop the original column
        encoded_df = pd.concat([encoded_df, dummies], axis=1).drop(columns=[col])

    return encoded_df


# Apply the function to both lists of columns
encoded_data = one_hot_encode_columns(data, columns_drop_first_true, columns_drop_first_false)

# Replace the original DataFrame with the encoded one
data = encoded_data

bin_columns = ['Have you been diagnosed with a mental health condition by a medical professional?',
               'Are you self-employed?', 'Is your employer primarily a tech company/organization?',
               'Do you have previous employers?',
               'Have you ever sought treatment for a mental health issue from a mental health professional?',
               'Inferred Tech Role']

# Initialize the encoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(data[bin_columns])

# Convert the encoded data to a DataFrame
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(bin_columns))

# Concatenate with the original data
data = pd.concat([data, one_hot_encoded_df], axis=1)

# Drop the original binary columns after encoding
data = data.drop(columns=bin_columns)
data['Inferred Tech Role_1.0'] = data['Inferred Tech Role_1.0'].astype(int)
data['Have you ever sought treatment for a mental health issue from a mental health professional?_1'] = data['Have you ever sought treatment for a mental health issue from a mental health professional?_1'].astype(int)
data['Do you have previous employers?_1'] = data['Do you have previous employers?_1'].astype(int)
data['Is your employer primarily a tech company/organization?_1'] = data['Is your employer primarily a tech company/organization?_1'].astype(int)
data['Are you self-employed?_1'] = data['Are you self-employed?_1'].astype(int)
data['Have you been diagnosed with a mental health condition by a medical professional?_1'] = data['Have you been diagnosed with a mental health condition by a medical professional?_1'].astype(int)



data.to_csv("processed_data.csv", index=False)

