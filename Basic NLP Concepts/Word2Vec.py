# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 00:49:05 2021

@author: HP
"""

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
paragraph = ''' India reacted to Kalam's death with an outpouring of grief; numerous tributes were paid to the former president across the nation and on social media.[96] The Government of India declared a seven-day state mourning period as a mark of respect.[97] President Pranab Mukherjee, Vice-President Hamid Ansari, Home Minister Rajnath Singh, and other leaders condoled the former President's demise.[87] Prime Minister Narendra Modi said "Kalam's death is a great loss to the scientific community. He took India to great heights. He showed the way."[98] Former Prime Minister Dr Manmohan Singh, who had served as prime minister under Kalam, said, "our country has lost a great human being who made phenomenal contributions to the promotion of self-reliance in defence technologies. I worked very closely with Dr. Kalam as prime minister and I greatly benefited from his advice as president of our country. His life and work will be remembered for generations to come."[99] ISRO chairman A. S. Kiran Kumar called his former colleague "a great personality and a gentleman", while former chairman G. Madhavan Nair described Kalam as "a global leader" for whom "the downtrodden and poor people were his priority. He always had a passion to convey what is in his mind to the young generation", adding that his death left a vacuum which none could fill.[100][101]

South Asian leaders expressed condolences and lauded the late statesman. The Bhutanese government ordered the country's flags to fly at half-staff to mourn Kalam's death and lit 1000 butter lamps in homage. Bhutanese Prime Minister Tshering Tobgay expressed deep sadness, saying Kalam "was a leader greatly admired by all people, especially the youth of India who have referred to him as the people's President".[102] Bangladesh Prime Minister Sheikh Hasina described Kalam as "a rare combination of a great statesman, acclaimed scientist, and a source of inspiration to the young generation of South Asia" and termed his death an "irreparable loss to India and beyond". Bangladesh Nationalist Party chief Khaleda Zia said "as a nuclear scientist, he engaged himself in the welfare of the people". Ashraf Ghani, the President of Afghanistan, called Kalam "an inspirational figure to millions of people," noting that "we have a lot to learn from his life". Nepalese Prime Minister Sushil Koirala recalled Kalam's scientific contributions to India: "Nepal has lost a good friend and I have lost an honoured and ideal personality." The President of Pakistan, Mamnoon Hussain, and Prime Minister of Pakistan Nawaz Sharif also expressed their grief and condolences on his death.[103][104][105] The President of Sri Lanka, Maithripala Sirisena, also expressed his condolences. "Dr. Kalam was a man of firm conviction and indomitable spirit, and I saw him as an outstanding statesman of the world. His death is an irreparable loss not only to India but to the entire world."[106] Maldivian President Abdulla Yameen and Vice-President Ahmed Adeeb condoled Kalam's death, with Yameen naming him as a close friend of the Maldives who would continue to be an inspiration to Indians and generations of South Asians. Former President Maumoon Abdul Gayoom, who had made an official visit to India during Kalam's presidency, termed his demise as a great loss to all of humankind.[107] The Commander-in-Chief of the Myanmar Armed Forces, Senior General Min Aung Hlaing, expressed condolences on behalf of the Myanmar government.[108] The Dalai Lama expressed his sadness and offered condolences and prayers, calling Kalam's death "an irreparable loss".[109]

Kathleen Wynne, the Premier of Ontario, which Kalam had visited on numerous occasions, expressed "deepest condolences ... as a respected scientist, he played a critical role in the development of the Indian space programme. As a committed educator, he inspired millions of young people to achieve their very best. And as a devoted leader, he gained support both at home and abroad, becoming known as 'the people's President'. I join our Indo–Canadian families, friends, and neighbours in mourning the passing of this respected leader."[110] United States President Barack Obama extended "deepest condolences to the people of India on the passing of former Indian President Dr. APJ Abdul Kalam", and highlighted his achievements as a scientist and as a statesman, notably his role in strengthening US–India relations and increasing space co-operation between the two nations. "Suitably named 'the People's President', Dr. Kalam's humility and dedication to public service served as an inspiration to millions of Indians and admirers around the world."[111] Russian President Vladimir Putin expressed sincere condolences and conveyed his sympathy and support "to the near and dear ones of the deceased leader, to the government, and entire people of India". He remarked on Kalam's outstanding "personal contribution to the social, economic, scientific, and technical progress of India and in ensuring its national security," adding that Kalam would be remembered as a "consistent exponent of closer friendly relations between our nations, who has done a lot for cementing mutually beneficial Russian–Indian cooperation."[112] Other international leaders—including former Indonesian president Susilo Bambang Yudhoyono, Malaysian Prime Minister Najib Razak, Singaporean Prime Minister Lee Hsien Loong, President of the United Arab Emirates Sheikh Khalifa bin Zayed Al Nahyan, and Vice-President and Prime Minister of the United Arab Emirates and emir of Dubai Sheikh Mohammed bin Rashid Al Maktoum—also paid tribute to Kalam.[113][114] In a special gesture, Secretary-General of the United Nations Ban Ki-moon visited the Permanent Mission of India to the UN and signed a condolence book. "The outpouring of grief around the world is a testament of the respect and inspiration he has garnered during and after his presidency. The UN joins the people of India in sending our deepest condolences for this great statesman. May he rest in peace and eternity", Ban wrote in his message.
'''


#Data Clining , preprocessing

text = re.sub(r'\[[0-9]*\]',' ',paragraph)  
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)
#text = re.sub('[.,]',' ',text)

#preparing the dataset

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] =[lemmatizer.lemmatize(word) for word in sentences[i] if word not in set(stopwords.words('english'))]


#training Word2Vec model

model = Word2Vec(sentences,min_count=2)

words = model.wv.get_vecattr

#Finding word vector
vector = model.wv['expressed']

#most simmilar words
similar = model.wv.most_similar('people')

