### **Competition Timeline**

- Challenge website open: 15/12/2024
- Release training cases: 15/03/2025
- Start of challenge - Training phase: 15/03/2025 - 15/08/2025
- Preliminary testing phase (10 submissions per team): 01/06/2025 -
  15/08/2025
- Advertisement of the challenge at ESTRO 2025: 04/05/2025
- Testing phase (2 submissions per team): 16/07/2023 - 15/08/2023
- Deadline for algorithm information form and LNCS format algorithm description: 01/09/2025
- Announcements and invitation to present: 09/09/2025
- Post-challenge phase: submission possible for a couple of years


![](https://rumc-gcorg-p-public.s3.amazonaws.com/i/2024/11/26/aa5be7c5-ac02-475a-a142-ac470c275a47.png)

------------------------------------------------------------------------

### **<u>Rules</u>** 📰

ENTRY INTO THIS CHALLENGE CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL
RULES.

Every participant must sign up for a
<a href="https://grand-challenge.org/verifications/" target="_blank">verified</a> Grand-Challenge account on
<a href="http://www.grand-challenge.org/" target="_blank">www.grand-challenge.org</a> and join the challenge to be
able to submit an algorithm.

#### **Methods**

Only fully automatic methods are allowed. Methods should be submitted as
specified on the
<a href="https://synthrad2025.grand-challenge.org/data/" target="_blank">submission page</a>.

Inference of submitted algorithms should run on an
<a href="https://aws.amazon.com/ec2/instance-types/g4/" target="_blank">AWS g4dn.2xlarge</a> instance using a single GPU with 16
GB RAM, 8 cores CPU, and 32 GB RAM. Maximum inference time for a single
case (one patient) should not exceed 1 sec per frame plus model and data loading time.

#### **One account per participant/team**

Each participant/team can only use one account to submit to the
challenge. Submissions from multiple accounts will result in disqualification. Teams
are limited to five participants.

#### **Use of other training data/pre-trained models**

TrackRAD2025 will provide training data. Participants are allowed to use additional data that is publicly available. Participants are also allowed to use publicly available pre-trained models. The use of publicly available data and models must be reported in the document describing the submitted method.	

#### **Code of the submitted algorithm**

The top five teams must disclose and openly share their code to allow for future re-use of their algorithms. While all other teams are strongly encouraged to do so, it is not mandatory. The code should be provided within 14 days of the announcement of the winning participant.

#### **Award eligibility**

Each team can comprise five participants, but the organizers reserve the right to reduce the number of co-authors of the top-performing teams to the challenge paper summarizing the results (see publication policy). Once a participant or a team submits, the submission or the team cannot withdraw from the challenge.

As further conditions for being awarded a prize, the teams must fulfill the following obligations:

- Present their method in person at the final event of the challenge at MICCAI 2025 if among the top five teams. 
- Submit a paper reporting the details of the methods in a short or long LNCS format, following the checklist provided on the submission page. Organizers reserve the right to exclude submissions lacking any of these reporting elements.
- Submit a form reporting the details of the algorithm after the testing phase submission has been completed, as the organizers will provide it.
- Sign and return all prize acceptance documents as may be required by the competition Sponsor/Organizers.
- Commit to citing the challenge report and data overview paper whenever submitting the developed method for scientific and non-scientific publications.
- The top five teams must disclose and openly share their code to allow for future re-use of their algorithms. While all other teams are strongly encouraged to do so, it is not mandatory. The code should be provided within 14 days of the announcement of the winning participant.



#### **Awards**

The results and winner will be announced publicly, and the top teams
will be invited to present their approach during the final MICCAI event.
Prizes will be awarded to the top five teams.

#### **Prizes** 🏆

The following prizes are awarded to the top five submissions:

- 1\. 1000,- USD
- 2\. 600,- USD
- 3\. 400,- USD
- 4\. 300,- USD
- 5\. 200,- USD

**Participation policy for organizers' institutes**  

Members of the organizers' institutes may participate if they are not
listed as organizers, contributors, or data providers, and they did not
co-author any publications with the organizers from 2023-09 to 2025-09.

#### **No private sharing outside teams**

Private sharing of code or data outside teams is prohibited.

#### **Data**

The dataset is released under a
<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC-BY(-NC) license</a> in .mha format.
The training data is available on Zenodo. Preliminary testing and final testing data is only available for evaluation via a submission (also possible post-challenge completion).

#### **Follow-up publication**

The TrackRAD2025 organizers will consolidate results and submit a
challenge paper to Medical Image Analysis or similar. The top ten teams in each
task will be invited to participate in the publication.

#### **Publishing the submitted method elsewhere**

Organizers and data providers may publish methods based on challenge
data after a 6-month embargo from the final MICCAI event. Participants may
submit their results elsewhere after the embargo, unless they cite the
overview paper, in which case no embargo applies.

#### **Other rules**

Once a participant or team submits, they cannot withdraw from the
challenge. Further rules are detailed on the challenge design page.