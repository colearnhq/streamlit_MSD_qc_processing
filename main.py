import streamlit as st
import datetime
import pandas as pd
import boto3
from io import StringIO
import string
import re
import logging
import collections

origin_bucket = st.secrets["origin_bucket"]
origin_prefix = st.secrets["origin_prefix"]

destination_bucket = st.secrets["destination_bucket"]
destination_prefix = st.secrets["destination_prefix"]
destination_file = destination_prefix + 'processed_file_id.csv'

client = boto3.client('s3',
                    aws_access_key_id = st.secrets["aws_access_key_id"],
                    aws_secret_access_key = st.secrets["aws_secret_access_key"]
                    )

# logging class
class TailLogHandler(logging.Handler):

    def __init__(self, log_queue):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))

# loggin class
class TailLogger(object):

    def __init__(self, maxlen):
        self._log_queue = collections.deque(maxlen=maxlen)
        self._log_handler = TailLogHandler(self._log_queue)

    def contents(self):
        return '\n'.join(self._log_queue)

    @property
    def log_handler(self):
        return self._log_handler

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache
def read_day(origin_key):
    csv_obj = client.get_object(Bucket=origin_bucket, Key=origin_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    
    return pd.read_csv(StringIO(csv_string))

st.set_page_config(page_title="MSD Data", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("MSD Data Cleaning")

# two columns for the start date and end date
col1, col2 = st.columns(2)
with col1:
    start_date_selector = st.date_input(
        label="Start Date",
        value=datetime.datetime.today() - datetime.timedelta(days=1),
        min_value=datetime.datetime.today() - datetime.timedelta(days=90),
        max_value=datetime.datetime.today() - datetime.timedelta(days=1),
        key='start_date'
        )

with col2:
    end_date_selector = st.date_input(
        label="End Date",
        value=datetime.datetime.today() - datetime.timedelta(days=1),
        min_value=datetime.datetime.today() - datetime.timedelta(days=90),
        max_value=datetime.datetime.today() - datetime.timedelta(days=1),
        key='end_date'
        )

# error correcting if user missinput between start date and end date
start_date = min([start_date_selector, end_date_selector])
end_date = max([start_date_selector, end_date_selector])

# downloading the data for each day 
date_range = pd.date_range(start_date, end_date, freq='d')
read_fail = []
raw_dataset = pd.DataFrame()
for date in date_range:
    origin_key = origin_prefix+str(date)[0:10]+'/000.csv'
    try:
        df = read_day(origin_key)
        raw_dataset = raw_dataset.append(df, ignore_index = True)
        print(f"{origin_key} success")
    except:
        st.warning(f"{origin_key} failed")
        print(f"{origin_key} failed")
        read_fail.extend(origin_key)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('Processing', key='process'):
        with st.spinner('Loading Files'):
            # load file from s3 
            obj = client.get_object(Bucket=destination_bucket, Key=destination_file)
            previous_file = pd.read_csv(obj['Body'])  # 'Body' is a key word

            all_file_ids = list(previous_file['file_id'].values)
            last_file_id = all_file_ids[-1]
            curr_id = last_file_id+1

            # logging initialization
            logger = logging.getLogger("__process__") # creating logging variable
            logger.setLevel(logging.DEBUG) # set the minimun level of loggin to DEBUG
            formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s") # logging format that will appear in the log file
            tail = TailLogger(10000) # amount of log that is saved
            log_handler = tail.log_handler # variable log handler
            log_handler.setFormatter(formatter) # set formatter to handler
            logger.addHandler(log_handler) # adding file handler to logger

            try:
                logger.info("QA User Billing Start")
                dataset = raw_dataset.copy()

                #data mapping and feedback labeling
                dataset['created_date'] = pd.to_datetime(dataset['created_date'])
                dataset['feedback_tags1'] = dataset['extras.tags'].astype(str).apply(remove_punctuations)
                dataset['extras.subject'] = dataset['extras.subject'].astype(str).apply(remove_punctuations)
                # dataset['extras.image_type'] = dataset['extras.image_type'].astype(str).apply(remove_punctuations)
                feedback_label = {'different' : 0, '25 Similar' : 1, '50 Similar' : 2, '75 Similar' : 3, '100 Similar' : 4, 'exact' : 5}
                dataset['feedback_tags'] = dataset['feedback_tags1'].map(feedback_label)

                undefined_user_list = ["VUEFIND_undefined", "VUEFIND_null"]
                undefined_user = dataset[dataset['qa_user_id'].isin(undefined_user_list)]
                dataset = dataset[~dataset['qa_user_id'].isin(undefined_user_list)]
                dataset = pd.concat([dataset, undefined_user])

                # copy tagging data from index = -1 to all index
                dataset['new_index'] = dataset['index']
                dataset['new_index'][dataset['index'] == -2] = 10

                tag_columns = ['extras.flagged', 'extras.sample_type', 'extras.image_type',
                    'extras.picture_taken', 'extras.subject', 'extras.section',
                    'extras.chapter', 'extras.topic', 'extras.not_found_in_index']

                dataset.sort_values(by = ['request_id', 'qa_user_id', 'new_index'], inplace = True)
                dataset[tag_columns] = dataset.groupby(['request_id', 'qa_user_id'])[tag_columns].fillna(method='ffill')

                dataset = dataset.drop('new_index', axis=1)

                # keeping later date data for same request_id and index for same qa user
                dataset1 = dataset.groupby(['request_id', 'qa_user_id', 'index'], group_keys=False, as_index=False).apply(lambda x: x.loc[x.created_date.idxmax()])

                total_qa_users = dataset1['qa_user_id'].nunique()
                unique_request_ids = dataset1['request_id'].nunique()
                index_wise_requests = dataset1[['index', 'request_id']].groupby('index').nunique()

                # Invalid data
                invalid = ['Not a relevant subject question', 'Invalid Question']
                words_re = re.compile("|".join(invalid))
                dataset1['is_valid'] = dataset1['extras.image_type'].apply(lambda x: False if words_re.search(str(x)) else True)
                invalid_requests = dataset1[dataset1['is_valid']==False]

                # Exact first position
                combined_invalid = invalid_requests.groupby('qa_user_id')['request_id'].apply(set).reset_index(name='request_list_invalid')
                dataset1 = pd.merge(dataset1, combined_invalid, how='left', on=['qa_user_id'])

                in_combined_invalid = []
                for i in dataset1.index:
                    in_combined_invalid += [True] if (pd.notnull(dataset1.loc[i, 'request_list_invalid']) and dataset1['request_id'][i] in dataset1['request_list_invalid'][i]) else [False]

                dataset1['in_combined_invalid'] = in_combined_invalid

                exact_first_position = dataset1[(dataset1['index']==0) & (dataset1['feedback_tags']==5) & (dataset1['in_combined_invalid']==False)]

                # tagged -2 index
                exact_invalid = pd.concat([exact_first_position, invalid_requests], axis=0)
                combined_exact_invalid = exact_invalid.groupby('qa_user_id')['request_id'].apply(set).reset_index(name='request_exact_invalid')
                dataset1 = pd.merge(dataset1, combined_exact_invalid, how='left', on=['qa_user_id'])

                in_combined_exact_invalid = []
                for i in dataset1.index:
                    in_combined_exact_invalid += [True] if (pd.notnull(dataset1.loc[i, 'request_exact_invalid']) and dataset1['request_id'][i] in dataset1['request_exact_invalid'][i]) else [False]

                dataset1['in_combined_exact_invalid'] = in_combined_exact_invalid

                feedback_list = dataset1[dataset1['index']>=0].groupby(['request_id', 'qa_user_id'])['feedback_tags'].apply(list).reset_index(name='feedback_list')
                neg2_requests = dataset1[(dataset1['index']==-2) & (dataset1['feedback_tags'] >= 4) & (dataset1['in_combined_exact_invalid'] == False)]
                neg2_requests = neg2_requests.merge(feedback_list, how = 'inner', on = ['request_id', 'qa_user_id'])
                neg2_requests['has_positive'] = neg2_requests['feedback_list'].apply(lambda x : True if any(i >=4 for i in x) else False)
                neg2_requests = neg2_requests[neg2_requests['has_positive']==False]

                # tagged all 10
                combined3 = pd.concat([exact_first_position, invalid_requests, neg2_requests], axis=0)
                combined3_req_id = combined3.groupby('qa_user_id')['request_id'].apply(set).reset_index(name='request_list_all3')
                dataset1 = pd.merge(dataset1, combined3_req_id, how='left', on=['qa_user_id'])

                in_combined_3 = []
                for i in dataset1.index:
                    in_combined_3 += [True] if (pd.notnull(dataset1.loc[i, 'request_list_all3']) and dataset1['request_id'][i] in dataset1['request_list_all3'][i]) else [False]

                dataset1['in_combined_3'] = in_combined_3

                all_10_tags = dataset1[dataset1['in_combined_3']==False]

                # final users with request count list
                exact_first = exact_first_position[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                exact_first['price'] = exact_first['request_id']*600
                exact_first.columns = ['qa_user_id','exact_first_position','exact_price']

                invalid_req = invalid_requests[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                invalid_req['price'] = invalid_req['request_id']*600
                invalid_req.columns = ['qa_user_id','invalid_requests','invalid_price']

                neg2_req = neg2_requests[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                neg2_req['price'] = neg2_req['request_id']*1800
                neg2_req.columns = ['qa_user_id','tagged_minus2','neg2_price']

                tagged_all_10 = all_10_tags[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                tagged_all_10['price'] = tagged_all_10['request_id']*1200
                tagged_all_10.columns = ['qa_user_id','tagged_all_10','all10_price']

                final_df = pd.merge(exact_first, invalid_req, how='outer', on=['qa_user_id'])
                final_df = pd.merge(final_df, neg2_req, how='outer', on=['qa_user_id'])
                final_df = pd.merge(final_df, tagged_all_10, how='outer', on=['qa_user_id'])
                final_df.fillna(0, inplace=True)
                final_df['total_count'] = final_df['exact_first_position'] + final_df['invalid_requests'] + final_df['tagged_minus2'] + final_df['tagged_all_10']
                final_df['total_price'] = final_df['exact_price'] + final_df['invalid_price'] + final_df['neg2_price'] + final_df['all10_price']

                invalid_user_mistakes = invalid_requests[invalid_requests['index']!=-1][['qa_user_id','request_id']].groupby('qa_user_id', as_index=False).nunique('request_id')
                invalid_user_mistakes['request_id'] = 'Found'
                invalid_user_mistakes.columns = ['qa_user_id','invalid_requests']

                exact_reqs = exact_first_position['request_id'].values
                exact_first_mistakes = dataset1[(dataset1['request_id'].isin(exact_reqs)) & (~dataset1['index'].isin([0,-1]))][['qa_user_id','request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                exact_first_mistakes['request_id'] = 'Found'
                exact_first_mistakes.columns = ['qa_user_id','exact_first_position']

                negative2 = neg2_requests['request_id'].values
                neg2_mistakes = dataset1[(dataset1['request_id'].isin(negative2)) & (dataset1['index'] >= 0) & (dataset1['feedback_tags'] >= 4)][['qa_user_id','request_id']].groupby('qa_user_id', as_index=False).nunique('request_id')
                neg2_mistakes['request_id'] = 'Found'
                neg2_mistakes.columns = ['qa_user_id','tagged_minus2']

                final_mistakes = pd.merge(invalid_user_mistakes, exact_first_mistakes, how='outer', on='qa_user_id')
                final_mistakes = pd.merge(final_mistakes, neg2_mistakes, how='outer', on='qa_user_id')
                
                logger.info("QA User Billing Finish")

                # s3 file uploading
                all_file_ids.append(curr_id)
                current_file = pd.DataFrame(all_file_ids, columns=['file_id'])

                output_file_names = [f'qa_user_request_count_{curr_id}.csv', 'processed_file_id.csv', f'users_not_following_rule_{curr_id}.csv']
                output_files = [final_df, current_file, final_mistakes]

                for i in range(3):
                    try:
                        with StringIO() as csv_buffer:
                            output_files[i].to_csv(csv_buffer, index=False)
                            output_file = destination_prefix + output_file_names[i]
                            print(output_file)
                            response = client.put_object(Bucket=destination_bucket, Key=output_file, Body=csv_buffer.getvalue())
                    except Exception as e:
                        print(e)
                
                # 
                st.text(f'File ID : {curr_id}')
                processed_user = len(final_df['qa_user_id'])
                st.text(f'Process User : {processed_user}')
                st.text(f'Error : {read_fail}')

                logger.info(f'File ID : {curr_id}')
                processed_user = len(final_df['qa_user_id'])
                logger.info(f'Process User : {processed_user}')
                logger.error(f'Error : {read_fail}')

            except Exception as e:
                st.error(f'Error Processing QA Billing : {e}')
                logger.error(f"Error Processing QA Billing : {e}")
            
            val_log = tail.contents() # extracting the log 

            # deleting all loggin variable for the current process
            log_handler.close()
            logging.shutdown()
            logger.removeHandler(log_handler)
            del logger, log_handler

            # saving the log file to S3
            try:
                log_filename = f"msd_data_log_{curr_id}.txt" # the name of the log file
                client.put_object(Bucket=destination_bucket, Key=destination_prefix + log_filename, Body=val_log)
                print(destination_prefix + log_filename)
                print(val_log)
            except Exception as e:
                print(e)

with col2:
    st.download_button('Download RAW', convert_df(raw_dataset), file_name='raw_msd.csv', mime='text/csv', key='raw')


# === DOWNLOAD SECTION ===
st.header("")
st.header("")
st.header("Download Processed File")
dwn_file_id = st.text_input("Type in file id to download deeplink processed file")
dwn_file_type = st.radio("Choose file to download", ['QA User Request Count', 'Users Not Following Rule'], index=0)

if dwn_file_id != "":
    try:
        if dwn_file_type == 'QA User Request Count':
            dwn_file_name = f'qa_user_request_count_{dwn_file_id}.csv'
        elif dwn_file_type == 'Users Not Following Rule':
            dwn_file_name = f'users_not_following_rule_{dwn_file_id}.csv'
        else:
            dwn_file_name = None
            raise Exception("Wrong Option")
            
        dwn_file = destination_prefix + dwn_file_name
        obj = client.get_object(Bucket= destination_bucket, Key= dwn_file)
        dwn_data = pd.read_csv(obj['Body']) # 'Body' is a key word
        csv = convert_df(dwn_data)

        st.download_button(
            label="Download",
            data=csv,
            file_name=dwn_file_name,
            mime='text/csv',
        )

    except:
        st.error("File ID not found in S3")