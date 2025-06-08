import pandas as pd
import re

# Assign a dataset to variable data

data = pd.read_csv('data/mental-health.csv', sep=',')
data.columns = data.columns.str.strip()

data_original = data.copy()

# Count number of duplicate rows
duplicate_count = data.duplicated().sum()

# Handling wrong data 'What is your age'

data = data[(data['What is your age?'] >= 18) & (data['What is your age?'] <= 70)]
data = data.reset_index(drop=True)


# Handling wrong data 'What is your gender'
def clean_gender(value):
    if pd.isnull(value):
        return 'No response'

    value = value.strip().lower()

    if value in ['male', 'm', 'man', 'cis male', 'cis male ', 'cis man', 'cisdude', 'mail', 'male ', 'male.',
                 'male (cis)', 'male/genderqueer', 'sex is male', 'male 9:1 female, roughly', 'dude']:
        return 'Male'

    if value in ['female', 'f', 'female ', 'woman', 'fem', 'fm',
                 'female (props for making this a freeform field, though)', 'female assigned at birth',
                 'female or multi-gender femme', 'cis female', 'cisgender female', 'cis-woman', 'i identify as female.',
                 'female/woman', 'female-bodied; no feelings about gender']:
        return 'Female'

    if 'trans' in value or 'nonbinary' in value or 'gender' in value or 'nb' in value or 'fluid' in value or 'queer' in value or 'unicorn' in value or 'androgynous' in value or 'bigender' in value or 'agender' in value or 'human' in value or 'other' in value:
        return 'Other'

    if "i'm a man" in value:
        return 'Male'

    if 'mtf' in value or 'transitioned' in value:
        return 'Other'

    if 'none of your business' in value:
        return 'Other'

    return 'Other'  # default fallback if unknown


# Apply the cleaning function
data['What is your gender?'] = data['What is your gender?'].apply(clean_gender)

# Adding values in the column 'If yes, what condition(s) have you been diagnosed with?'
col_if_yes_diagnosed = 'If yes, what condition(s) have you been diagnosed with?'
col_currently_have_disorder = 'Do you currently have a mental health disorder?'

# For rows where they said Yes but diagnosis is missing, fill with 'Diagnosis not provided'
data.loc[
    (data[col_currently_have_disorder] == 'Yes') & (data[col_if_yes_diagnosed].isnull()),
    col_if_yes_diagnosed
] = 'Diagnosis not provided'

data.loc[
    data[col_if_yes_diagnosed].isnull() &
    (data[col_currently_have_disorder] == 'Maybe'),
    col_if_yes_diagnosed
] = "Don't know"
data.loc[
    data[col_if_yes_diagnosed].isnull() &
    (data[col_currently_have_disorder] == 'No'),
    col_if_yes_diagnosed
] = "Not applicable"

# Finished cleaning 'If yes, what condition(s) have you been diagnosed with?'


# Adding values in the column 'col_less_likely_to_reveal' and replacing NaN values(float) with 'No response'(str)
col_obs_unsupportive_response = ('Have you observed or experienced an unsupportive or badly handled response to a '
                                 'mental health issue in your current or previous workplace?')
col_less_likely_to_reveal = ('Have your observations of how another individual who discussed a mental health disorder '
                             'made you less likely to reveal a mental health issue yourself in your current workplace?')

data.loc[
    data[col_less_likely_to_reveal].isnull() &
    data[col_obs_unsupportive_response].isin(['Yes, I experience', 'Yes, I observed']),
    col_less_likely_to_reveal
] = 'Yes'

data.loc[:, col_less_likely_to_reveal] = data.loc[:, col_less_likely_to_reveal].fillna('No response')

# Replacing NaN values from the col_obs_unsupportive_response with 'No response'


data.loc[:, col_obs_unsupportive_response] = data.loc[:, col_obs_unsupportive_response].fillna('No response')
#  Finished cleaning 'col_less_likely_to_reveal'

# Deleting columns that contain mostly more than 70% of missing values
data.columns = data.columns.str.strip()

col_have_medical_cov = ('Do you have medical coverage (private insurance or state-provided) '
                        'which includes treatment of  mental health issues?')
col_know_local_resources = 'Do you know local or online resources to seek help for a mental health disorder?'
col_reveal_to_business_contacts = ('If you have been diagnosed or treated for a mental health disorder,'
                                   ' do you ever reveal this to clients or business contacts?')
col_reveal_impacted_negatively = ('If you have revealed a mental health issue to a client or business contact,'
                                  ' do you believe this has impacted you negatively?')
col_reveal_to_coworkers = ('If you have been diagnosed or treated for a mental health disorder, do you ever '
                           'reveal this to coworkers or employees?')
col_reveal_coworker_impacted_negatively = ('If you have revealed a mental health issue to a '
                                           'coworker or employee, do you believe this has impacted you negatively?')
col_productivity_affected = 'Do you believe your productivity is ever affected by a mental health issue?'
col_time_affected_mental = ('If yes, what percentage of your work time '
                            '(time performing primary or secondary job functions) is affected by a mental health issue?')
col_if_maybe = 'If maybe, what condition(s) do you believe you have?'

columns_to_drop = [col_have_medical_cov, col_know_local_resources, col_reveal_to_business_contacts,
                   col_reveal_impacted_negatively,
                   col_reveal_to_coworkers, col_reveal_coworker_impacted_negatively, col_productivity_affected,
                   col_time_affected_mental, col_if_maybe]
data = data.drop(columns=columns_to_drop)
# Finished deleting columns that mostly contain missing values


cols_to_drop = [
    'What US state or territory do you live in?',
    'What US state or territory do you work in?'
]

data = data.drop(columns=cols_to_drop)

# 'If so, what condition(s) were you diagnosed with?'. the answer 'So' does not refer to the
# definite answer, it is also not clear to what column it is correlated.
# 'Do you know the options for mental health care available under your employer-provided coverage?'
# It is too specific
# Delete the column 'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?'
# This column is repeated
cols_to_drop = [
    'If so, what condition(s) were you diagnosed with?',
    'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?',
    'Do you know the options for mental health care available under your employer-provided coverage?'
]
data = data.drop(columns=cols_to_drop)


# Creating a new column Inferred Tech Role
def is_tech_role(position):
    tech_keywords = ['back-end developer', 'dev evangelist/advocate',
                     'devops/sysadmin', 'front-end developer', 'one-person shop']
    if pd.isnull(position):
        return False
    roles = [r.strip().lower() for r in position.split('|')]
    return any(role in tech_keywords for role in roles)


data['Inferred Tech Role'] = data['Which of the following best describes your work position?'].apply(is_tech_role)
data['Inferred Tech Role'] = data['Inferred Tech Role'].map({True: 1, False: 0})

col_is_tech = 'Is your primary role within your company related to tech/IT?'
data[col_is_tech] = data[col_is_tech].fillna(data['Inferred Tech Role'])

# Finished adding values into the column 'Inferred Tech Role'

data[col_is_tech] = data[col_is_tech].astype('Int64')
data['Inferred Tech Role'] = data['Inferred Tech Role'].astype('Int64')

# Here we see the mismatch between the answer of respondent and his/her position.
# Some of the respondents answered they work in tech, but their positions are not tech-related
# It is therefore decided to trust the answers of participants and their answers are taken as a true.

# Find difference
diff = data[col_is_tech] != data['Inferred Tech Role']

# Correct the Inferred Tech Role where mismatch
data.loc[diff, 'Inferred Tech Role'] = data.loc[diff, col_is_tech]
# Recalculate the difference after updating
diff = data[col_is_tech] != data['Inferred Tech Role']

# So now we have 2 identical columns, col_is_tech can be deleted
data = data.drop(columns=col_is_tech)
data = data.drop(columns=['Which of the following best describes your work position?'])

# So now we have a column 'Inferred Tech Role'.

# Adding missing values to the column 'If maybe, what condition(s) do you believe you have?', this column is
# correlated to the column 'Do you currently have a mental health disorder?'

col_currently_have_disorder = 'Do you currently have a mental health disorder?'

# Adding missing values to the column 'Why or why not'

# Rename the wrong column name to the correct one
data = data.rename(columns={'Why or why not?.1': 'Why or why not bring up with a potential employer in an interview'
                                                 '(mental health issue)?'})
data = data.rename(columns={'Why or why not?': 'Why or why not bring up with a potential employer in an interview'
                                               '(physical health issue)?'})


def clean_text(text):
    if pd.isnull(text):
        return 'no response'
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()


data['Why or why not bring up with a potential employer in an interview'
     '(mental health issue)_cleaned?'] = data['Why or why not bring up with a potential employer in an interview'
                                              '(mental health issue)?'].apply(clean_text)
data['Why or why not bring up with a potential employer in an interview'
     '(physical health issue)_cleaned?'] = data['Why or why not bring up with a potential employer in an interview'
                                                '(physical health issue)?'].apply(clean_text)
data = data.drop(columns=[
    'Why or why not bring up with a potential employer in an interview(mental health issue)?',
    'Why or why not bring up with a potential employer in an interview(physical health issue)?'
])


# Finished adding values to the column 'Why or why not'


# Adding values to the column 'How many employees does your company or organization have?'

col_self_empl = 'Are you self-employed?'
col_how_many_empl = 'How many employees does your company or organization have?'

data.loc[
    data[col_how_many_empl].isnull() &
    (data[col_self_empl] == 1),
    col_how_many_empl
] = '0'


# Create a mapping dictionary
employee_size_map = {
    '1-5': 0,
    '6-25': 0,
    '26-100': 0,
    '100-500': 1,
    '500-1000': 1,
    'More than 1000': 1,
    '0': 2
}

# Apply the mapping
data[col_how_many_empl] = data[col_how_many_empl].map(employee_size_map)

# Adding values to the column 'Is your employer primarily a tech company/organization?'

col_is_empl_tech = 'Is your employer primarily a tech company/organization?'

data.loc[
    data[col_is_empl_tech].isnull() &
    (data[col_self_empl] == 1) &
    (data['Inferred Tech Role'] == 1),
    col_is_empl_tech
] = 1

data.loc[
    data[col_is_empl_tech].isnull() &
    (data[col_self_empl] == 1) &
    (data['Inferred Tech Role'] == 0),
    col_is_empl_tech
] = 0
data[col_is_empl_tech] = data[col_is_empl_tech].astype(int)
# Finished adding values to the column 'Is your employer primarily a tech company/organization?'

# Adding values to the column 'Does your employer provide mental health benefits as part of healthcare coverage?'

# Define your column names
coverage_provided_by_empl = 'Does your employer provide mental health benefits as part of healthcare coverage?'

# Fill missing mental health coverage for self-employed people
data.loc[
    data[coverage_provided_by_empl].isnull() & (data[col_self_empl] == 1),
    coverage_provided_by_empl
] = 'No'

# Simple replace
data[coverage_provided_by_empl] = data[coverage_provided_by_empl].replace('Not eligible for coverage / N/A', 'No')

# Adding values to the column 'If a mental health issue prompted you to request a medical leave from work,
# asking for that leave would be:'

leave = 'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'

data.loc[
    data[leave].isnull() & (data[col_self_empl] == 1), leave] = 'Not applicable'

# Adding values to the column Is your anonymity protected if you choose to take advantage of mental health or
# substance abuse treatment resources provided by your employer?

anonymity = ('Is your anonymity protected if you choose to take advantage of mental health or substance abuse '
             'treatment resources provided by your employer?')

data.loc[
    data[anonymity].isnull() & (data[col_self_empl] == 1), anonymity] = 'Not applicable'
# Finished adding values in the column anonymity

# Adding values to the column 'Does your employer offer resources to learn more about mental health concerns and
# options for seeking help?'

offered_recources = ('Does your employer offer resources to learn more about mental health concerns and options for '
                     'seeking help?')
data.loc[
    data[offered_recources].isnull() & (data[col_self_empl] == 1), offered_recources] = 'Not applicable'
# Finished adding values to the column offered recources

# Adding values to the column ''Has your employer ever formally discussed mental health (for example, as part of a
# wellness campaign or other official communication)?

wellness_campaign = ("Has your employer ever formally discussed mental health (for example, as part of a wellness "
                     "campaign or other official communication)?")

data.loc[
    data[wellness_campaign].isnull() & (data[col_self_empl] == 1), wellness_campaign] = 'Not applicable'
# Finished adding values to the column wellness campaign

# Adding values to the column 'Do you think that discussing a mental health disorder with your employer would have
# negative consequences?'  and
# 'Do you think that discussing a physical health issue with your employer would have negative consequences?'


discussing_mental_negative_conseq = ('Do you think that discussing a mental health disorder with your employer would '
                                     'have negative consequences?')
discussing_physical_negative_conseq = ("Do you think that discussing a physical health issue with your employer"
                                       " would have negative consequences?")
data.loc[
    data[discussing_mental_negative_conseq].isnull() & (data[col_self_empl] == 1), discussing_mental_negative_conseq
] = 'Not applicable'

data.loc[
    data[discussing_physical_negative_conseq].isnull() & (data[col_self_empl] == 1), discussing_physical_negative_conseq
] = 'Not applicable'

# Finished adding values to the columns discussing_mental_negative_conseq and discussing_physical_negative_conseq

# Adding values to the column 'Would you feel comfortable discussing a mental health disorder with your coworkers?'
# and 'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?' and
# 'Do you feel that your employer takes mental health as seriously as physical health?' and
# 'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?'

comfort_discussing_coworkers = 'Would you feel comfortable discussing a mental health disorder with your coworkers?'

comfort_discussing_supervisor = ('Would you feel comfortable discussing a mental health disorder '
                                 'with your direct supervisor(s)?')

empl_takes_serious_mental_physical = ('Do you feel that your employer takes mental health as seriously'
                                      ' as physical health?')

observed_negative_conseq = ('Have you heard of or observed negative consequences for co-workers'
                            ' who have been open about mental health issues in your workplace?')

data.loc[
    data[comfort_discussing_coworkers].isnull() & (data[col_self_empl] == 1), comfort_discussing_coworkers
] = 'Not applicable'

data.loc[
    data[comfort_discussing_supervisor].isnull() & (data[col_self_empl] == 1), comfort_discussing_supervisor
] = 'Not applicable'

data.loc[
    data[empl_takes_serious_mental_physical].isnull() & (data[col_self_empl] == 1), empl_takes_serious_mental_physical
] = 'Not applicable'

data.loc[
    data[observed_negative_conseq].isnull() & (data[col_self_empl] == 1), observed_negative_conseq
] = 'Not applicable'

# Finished adding values to the columns: comfort_discussing_coworkers, comfort_discussing_supervisor,
# empl_takes_serious_mental_physical, observed_negative_conseq

# Adding values to the columns
# Have your previous employers provided mental health benefits?
# Were you aware of the options for mental health care provided by your previous employers?
# Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?
# Did your previous employers provide resources to learn more about mental health issues and how to seek help?
# Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?
# Do you think that discussing a mental health disorder with previous employers would have negative consequences?
# Do you think that discussing a physical health issue with previous employers would have negative consequences?
# Would you have been willing to discuss a mental health issue with your previous co-workers?
# Did you feel that your previous employers took mental health as seriously as physical health?
# Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?

having_previous_empl = 'Do you have previous employers?'

col_benefits_previous_empl = 'Have your previous employers provided mental health benefits?'

data.loc[
    data[col_benefits_previous_empl].isnull() & (data[having_previous_empl] == 0), col_benefits_previous_empl
] = 'No previous employer'

aware_options_mental_prev_empl = ('Were you aware of the options for mental health care provided '
                                  'by your previous employers?')

data.loc[
    data[aware_options_mental_prev_empl].isnull() & (data[having_previous_empl] == 0), aware_options_mental_prev_empl
] = 'No previous employer'

mapping = {'Yes, I was aware of all of them': 'Aware',
           'I was aware of some': 'Aware',
           'No, I only became aware later': 'Not Aware',
           'N/A (not currently aware)': 'Not Aware',
           'No previous employer': 'No History'}
data[aware_options_mental_prev_empl] = data[aware_options_mental_prev_empl].map(mapping)

discussing_prev_empl_wellness_campaign = ('Did your previous employers ever formally '
                                          'discuss mental health (as part of a wellness campaign '
                                          'or other official communication)?')
data.loc[
    data[discussing_prev_empl_wellness_campaign].isnull() & (
            data[having_previous_empl] == 0), discussing_prev_empl_wellness_campaign
] = 'No previous employer'

provided_resources_prev_empl = ('Did your previous employers provide resources to '
                                'learn more about mental health issues and how to seek help?')

data.loc[
    data[provided_resources_prev_empl].isnull() & (data[having_previous_empl] == 0), provided_resources_prev_empl
] = 'No previous employer'

anonymity_protected_prev_empl = ('Was your anonymity protected if '
                                 'you chose to take advantage of mental health or substance '
                                 'abuse treatment resources with previous employers?')
data.loc[
    data[anonymity_protected_prev_empl].isnull() & (data[having_previous_empl] == 0), anonymity_protected_prev_empl
] = 'No previous employer'

discussing_mental_neg_prev_empl = ('Do you think that discussing a mental health disorder'
                                   ' with previous employers would have negative consequences?')
data.loc[
    data[discussing_mental_neg_prev_empl].isnull() & (data[having_previous_empl] == 0), discussing_mental_neg_prev_empl
] = 'No previous employer'

discussing_physical_neg_prev_empl = ('Do you think that discussing a physical health '
                                     'issue with previous employers would have negative consequences?')
data.loc[
    data[discussing_physical_neg_prev_empl].isnull() & (
            data[having_previous_empl] == 0), discussing_physical_neg_prev_empl
] = 'No previous employer'

discussing_mental_prev_coworkers = ('Would you have been willing to discuss a mental health'
                                    ' issue with your previous co-workers?')

data.loc[
    data[discussing_mental_prev_coworkers].isnull() & (
            data[having_previous_empl] == 0), discussing_mental_prev_coworkers
] = 'No previous coworkers'

take_serious_mental_physical_prev_empl = ('Did you feel that your previous employers '
                                          'took mental health as seriously as physical health?')
data.loc[
    data[take_serious_mental_physical_prev_empl].isnull() & (
            data[having_previous_empl] == 0), take_serious_mental_physical_prev_empl
] = 'No previous employer'

observ_neg_conseq_prev_empl = ('Did you hear of or observe negative consequences for '
                               'co-workers with mental health issues in your previous workplaces?')
data.loc[
    data[observ_neg_conseq_prev_empl].isnull() & (data[having_previous_empl] == 0), observ_neg_conseq_prev_empl
] = 'No previous workplace'

# Finished adding values to the list of columns


def clean_conditions(text):
    if pd.isnull(text) or text is None:
        return ''

    # Convert to lowercase for uniformity
    text = text.lower()

    # Correct common multi-word conditions using regex to capture variations
    text = re.sub(r'obsessive[\s\-]?compulsive', 'obsessive-compulsive', text)
    text = re.sub(r'pdd[\s\-]?nos', 'pdd-nos', text)
    text = re.sub(r'attention[\s\-]?deficit[\s\-]?hyperactivity[\s\-]?disorder',
                  'attention-deficit-hyperactivity-disorder', text)
    text = re.sub(r'post[\s\-]?traumatic[\s\-]?stress[\s\-]?disorder', 'ptsd', text)
    text = re.sub(r'generalized[\s\-]?anxiety[\s\-]?disorder', 'generalized-anxiety-disorder', text)
    text = re.sub(r'bipolar[\s\-]?disorder', 'bipolar-disorder', text)

    # 1: Remove any text inside parentheses (subcategories)
    text = re.sub(r'\([^)]*\)', '', text)

    # 2: Remove any non-alphabetic characters except '|' and '-'
    text = re.sub(r'[^a-z|\s\-]', '', text)

    # 3: Normalize spaces and remove extra pipes
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\|+', '|', text).strip('|')

    return text


# Apply the clean function to the specific column
data[col_if_yes_diagnosed] = data[col_if_yes_diagnosed].apply(clean_conditions)


def concatenate_conditions(text):
    if pd.isnull(text) or text is None:
        return ''

    # Split the text by the '|' separator to isolate each condition
    conditions = [cond.strip() for cond in text.split('|') if cond.strip()]

    # Replace spaces within each condition with a hyphen
    processed_conditions = ['-'.join(cond.split()) for cond in conditions]

    # Rejoin the processed conditions with ' | ' to keep multi-label structure
    return ' | '.join(processed_conditions)


# Apply the function to the specific column
data[col_if_yes_diagnosed] = data[col_if_yes_diagnosed].apply(concatenate_conditions)

mapping_dict = {
    'add': 'attention-deficit-disorder',
    'asperges': 'asperger-syndrome',
    'autism-spectrum-disorder': 'autism',
    'combination-of-physical-impairment-with-a-possibly-mental-one': 'physical-mental-impairment',
    'diagnosis-not-provided': 'unknown',
    'dont-know': 'unknown',
    'gender-dysphoria': 'gender-identity-disorder',
    'i-havent-been-formally-diagnosed-so-i-felt-uncomfortable-answering-but-social-anxiety-and-depression': 'anxiety-disorder|depression',
    'not-applicable': 'healthy',
    'pdd-nos': 'pervasive-developmental-disorder',
    'ptsd': 'post-traumatic-stress-disorder'
}


def map_conditions(text):
    if pd.isnull(text) or text is None:
        return 'unknown'

    # Split the multi-label text by '|'
    conditions = [cond.strip() for cond in text.split('|') if cond.strip()]

    # Map each condition using the dictionary
    mapped_conditions = [mapping_dict.get(cond, cond) for cond in conditions]

    # Rejoin the mapped conditions using ' | '
    return ' | '.join(mapped_conditions)


# Apply the mapping function to the column
data[col_if_yes_diagnosed] = data[col_if_yes_diagnosed].apply(map_conditions)

# Answers in this column don't logically match
data = data.drop(columns='Would you have been willing to discuss a mental health issue with your previous co-workers?')
# Save the cleaned and encoded data to CSV for inspection
data.to_csv("cleaned_data.csv", index=False)
