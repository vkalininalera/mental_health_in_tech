import pandas as pd
import matplotlib.pyplot as plt

# Read summary if needed
summary = pd.read_csv("best_cluster_summary.csv", index_col=0)

# Define your feature groups
aggregated_groups = {
    "No Support Awareness": [
        "Does_your_employer_offer_resources_to_learn_more_about_mental_health_concerns_and_options_for_seeking_help?_No"],

    "Support awareness": [
        "Did_your_previous_employers_provide_resources_to_learn_more_about_mental_health_issues_and_how_to_seek_help?_Yes",
        "Were_you_aware_of_the_options_for_mental_health_care_provided_by_your_previous_employers?_Yes",
        "Have_your_previous_employers_provided_mental_health_benefits?_Yes"],

    "Openness and psychological safety": [
        "Would_you_bring_up_a_mental_health_issue_with_a_potential_employer_in_an_interview?_Yes",
        "Would_you_be_willing_to_bring_up_a_physical_health_issue_with_a_potential_employer_in_an_interview?_Yes",
        "Would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisor(s)?_Yes",
        "Do_you_think_that_team_members/coworkers_would_view_you_more_negatively_if_they_knew_you_suffered_from_a_mental_health_issue?_No",
        "Do_you_think_that_discussing_a_physical_health_issue_with_previous_employers_would_have_negative_consequences?_No",
        "Did_you_hear_of_or_observe_negative_consequences_for_co-workers_with_mental_health_issues_in_your_previous_workplaces?_No"
        ],

    "Work and live in Europe": ["What_country_do_you_work_in?_Europe",
                                "What_country_do_you_live_in?_Europe"],

    "Reticence": ["Would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_coworkers?_No",
                  "Would_you_feel_comfortable_discussing_a_mental_health_disorder_with_your_direct_supervisor(s)?_No"
                  "Would_you_be_willing_to_bring_up_a_physical_health_issue_with_a_potential_employer_in_an_interview?_No"
                  ],

    "Work in tech": ["Is your employer primarily a tech company/organization?_1",
                     "Inferred Tech Role_1.0"],

    "Remote Work": [
        "Do_you_work_remotely?_Sometimes",
        "Do_you_work_remotely?_Always"],


    "Experience with Stigma": ["Have_you_observed_or_experienced_an_unsupportive_or_badly_handled_response_to_a_mental_health_issue_in_your_current_or_previous_workplace?_Yes",
                               "Do_you_think_that_discussing_a_mental_health_disorder_with_previous_employers_would_have_negative_consequences?_Yes"],

    "Work in small company till 100": ["How_many_employees_does_your_company_or_organization_have?_0"],

    "Experienced employer": ["Do you have previous employers?_1",]

}

# Step 1: Build dictionary of averaged values
aggregated_data = {}

for group, features in aggregated_groups.items():
    existing = [f for f in features if f in summary.index]
    if existing:
        aggregated_data[group] = summary.loc[existing].mean()
# Add Age as its own group
if "What is your age?" in summary.index:
    aggregated_data["Average Age"] = summary.loc["What is your age?"]

# Step 2: Convert to DataFrame
aggregated_df = pd.DataFrame(aggregated_data).T  # Groups as rows

# Step 3: Plot
aggregated_df.T.plot(kind='bar', figsize=(10, 6), colormap='tab20')
plt.title("Aggregated Cluster Summary by Thematic Group")
plt.ylabel("Average Percentage")
plt.xlabel("Cluster")
plt.xticks(rotation=0)
plt.legend(title="Feature Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
