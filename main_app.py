import streamlit as st

#--- Replicate API Settings & llama2
#test_secret = st.secrets['test']
st.write("Hello World")
test_secret = st.secrets["REPLICATE_API_TOKEN"]
st.write(test_secret)

from time import sleep
import json
from matcher import load_student_data, load_universities_database, match
from openai import OpenAI
import replicate
import os

#--- Replicate API Settings & llama2
#test_secret = st.secrets['test']
st.write("Hello World")
test_secret = st.secrets["db_username"]
st.write(test_secret)



# https://uniguide.streamlit.app/

#------------------------------------------------------- Class & Functions -------------------------------------------------------
class Assistant():
    def __init__(self):
        self.messages = []
        self.role = "assistant"
        self.unimatch_questions = {
            1: "Tell me a little bit more about your hobbies!",
            2: "What fields in the school are you interested in?",
            3: "What is your budget for studying?"
        }
        self.unimatch_on = False
        self.unibuddy_on = False
        self.last_user_reply = ""
        self.user_replies_counter = 0
            

    def print_and_add_message(self, content):
        sleep(time_sleep_fast)
        #--- Print message
        with st.chat_message(self.role):
            st.markdown(content)

        #--- Add to history
        self.messages.append({"role": self.role, "content": content})
        st.session_state.messages.append({"role": self.role, "content": content})
    
    def unimatch_question(self):
        self.print_and_add_message(self.unimatch_questions[self.user_replies_counter])

    def set_unimatch_on(self, value):
        self.unimatch_on = value
    def set_unibuddy_on(self, value):
        self.unibuddy_on = value
    def set_last_user_reply(self, value):
        self.last_user_reply = value
    def update_user_replies_counter(self):
        #--- load the json file
        with open("cache-data.json") as f:
            data = json.load(f)

        #--- update the user_replies_counter
        self.user_replies_counter = data["user_replies_counter"]
        self.user_replies_counter += 1

        #--- update the json file
        data["user_replies_counter"] = self.user_replies_counter
        with open("cache-data.json", "w") as f:
            json.dump(data, f)

    def check_finished_questions(self):
        #--- get matching_done from the cache
        with open("cache-data.json") as f:
            data = json.load(f)
        matching_done = data["matching_done"]
        if self.user_replies_counter > len(self.unimatch_questions) and matching_done == False:
            self.print_and_add_message("Great! I have all the information I need. Let me find the best university for you! 🎓")
            self.unimatch_on = False
            self.user_replies_counter = 0
            #--- update the json file
            with open("cache-data.json") as f:
                data = json.load(f)
            data["user_replies_counter"] = self.user_replies_counter
            with open("cache-data.json", "w") as f:
                json.dump(data, f)
            return True

    def get_last_user_reply(self):
        return self.last_user_reply

class User():
    def __init__(self):
        self.messages = []
        self.role = "user"

    def print_and_add_message(self, content):
        sleep(time_sleep_fast)
        #--- Print message
        with st.chat_message(self.role):
            st.markdown(content)

        #--- Add to history
        self.messages.append({"role": self.role, "content": content})
        st.session_state.messages.append({"role": self.role, "content": content})

    def update_user_profile(self):
        #--- load the json file
        with open("cache-data.json") as f:
            data = json.load(f)
        data["user_profile"] = data["user_profile"] + self.messages[-1]["content"] + " | "

        #--- update the json file
        with open("cache-data.json", "w") as f:
            json.dump(data, f)

    
    def get_last_reply(self):
        return self.messages[-1]["content"]

class Llama2():
    def __init__(self):
        #--- Settings
        self.temperature = 0.1
        self.top_p = 0.9
        self.max_length = 256

        #--- Models
        self.llm_7b = "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea"
        self.llm_13b = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

        #--- Instructions
        self.instructions = [
            "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.",
            "Always give short answers and do not provide too much information.",
            "You only answer requests related to universities. If the topic is not this, answer: 'I'm not trained with data not related to universities.'",
        ]

    def generate_llama2_response(self, prompt_input, model = "llm_13b"):
        #--- Select model
        if model == "llm_13b": model = self.llm_13b
        if model == "llm_7b": model = self.llm_7b
        
        #--- Add initial instructions
        for instruction in self.instructions:
            string_dialogue = instruction + "\n\n"

        #--- Add the dialogue history
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        
        #--- Run the model
        output = replicate.run(model, 
                            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: "})
        return output

    def give_profile_overview(self, loaded_profile, best_university):
        #--- Give an overview of the user profile and the best university, and why it is the best match
        overview_profile = f"Why I'm good match with {best_university}? (answer only for the best university)"


        return self.generate_llama2_response(overview_profile)

st.title("UniGuide Chat App")
# ------------------------------------------------------- SETUP -------------------------------------------------------
#--- Initialize Agents
assistant = Assistant()
user = User()

time_sleep_fast = 0.25
time_sleep_longer = 1

#--- OpenAI API Settings
# Set OpenAI API key from Streamlit secrets
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
#if "openai_model" not in st.session_state:
#    st.session_state["openai_model"] = "gpt-3.5-turbo"

replicate_api = st.secrets['REPLICATE_API_TOKEN']
os.environ['REPLICATE_API_TOKEN'] = replicate_api
llama2 = Llama2()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Initial Messages ----
with st.chat_message("assistant"):
    st.markdown("Welcome to the complete experience of UniGuide Chatbot 👋!")
    st.markdown("I am here to help you find the best university for you and also to explore more about the universities information! 📖")
    st.markdown("Please, if you want to find the perfect match for you, type 'UniMatch'. If you want to discover more about the universities, type 'UniBuddy'")

#=======================================================
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
#=======================================================


if prompt := st.chat_input("What is up?"):
    user.print_and_add_message(prompt)

    # ------------------------------------------------------- UniMtach -------------------------------------------------------
    #========================= Introduction ==========================
    if prompt == "UniMatch" or prompt == "unimatch":
        assistant.print_and_add_message("Great! Let's find the perfect match for you! 🎓")
        assistant.print_and_add_message("Please, answer the following questions to help me find the best university for you:")

        #--- Update status unimatch
        assistant.set_unimatch_on(True)
        
        #--- Update matching_done to False
        with open("cache-data.json") as f:
            data = json.load(f)
        data["matching_done"] = False
        data["user_profile"] = ""
        data["user_replies_counter"] = 0
        with open("cache-data.json", "w") as f:
            json.dump(data, f)

    if prompt == "UniBuddy" or prompt == "unibuddy":
        assistant.print_and_add_message("Great! Let's explore more about the universities! 📚")
        #--- Update status unibuddy
        assistant.set_unibuddy_on(True)
        #--- Update matching_done to True
        with open("cache-data.json") as f:
            data = json.load(f)
        data["matching_done"] = True
        with open("cache-data.json", "w") as f:
            json.dump(data, f)

    #================================================================

    #========================= CHECKING ==========================
    if assistant.get_last_user_reply() != user.get_last_reply():
        #--- Update user replies counter
        assistant.update_user_replies_counter()
        assistant.set_last_user_reply(user.get_last_reply())
        assistant.unimatch_on = True

        #--- Update user profile
        user.update_user_profile()
    else:
        assistant.unimatch_on = False
    #--- check if matching is done on the cache
    with open("cache-data.json") as f:
        data = json.load(f)
    if data["matching_done"] == True:
        assistant.unimatch_on = False
        assistant.unibuddy_on = True
    #================================================================
    
    #============================== MATCHING ==============================
    if assistant.check_finished_questions():
        #--- load user profile from json
        student_data = load_student_data()
    
        #--- load universities data from json
        universities_data, universities_names = load_universities_database()

        #--- Match the student data with the universities data
        universities_similarities = match(student_data, universities_data, universities_names)
        # universities_similarities = {uniA: 0. , uniB: 0. , uniC: 0.}
        #--- Sort the universities based on the similarities
        universities_similarities = dict(sorted(universities_similarities.items(), key=lambda item: item[1], reverse=True))

        #------- Printing the Results -------
        with st.spinner("Thinking..."):
            sleep(3)
            assistant.print_and_add_message(f"Based on your answers, I found the best university for you are 🎉")
        
        # Print the 5 best universities and show the similarities
        top_5_universities = list(universities_similarities.items())[:5]
        for uni, sim in top_5_universities:
            assistant.print_and_add_message(f"{uni} with similarity of {int(sim*10000)/100}%")

        # --- Print a bar chart with the similarities and universities
        st.bar_chart(universities_similarities)

        #--- Clean User Profile after match
        with open("cache-data.json") as f:
            data = json.load(f)
        data["user_profile"] = ""
        with open("cache-data.json", "w") as f:
            json.dump(data, f)

        #--- Update the json file with matching done
        with open("cache-data.json") as f:
            data = json.load(f)
        data["matching_done"] = True
        with open("cache-data.json", "w") as f:
            json.dump(data, f)
        
        #---- User Profile Overview ----
        # Best University
        best_university = top_5_universities[0][0]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llama2.give_profile_overview(student_data, best_university)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
        

    #=====================================================================

    #============================== QUESTION ==============================
    if assistant.unimatch_on:
        assistant.unimatch_question()
    #=====================================================================

    
    # ------------------------------------------------------- UniBuddy -------------------------------------------------------


    if assistant.unibuddy_on:

        # TODO:
        # - Test API from chatgpt
        # - give an overview of the user profile and the best university, and why it is the best match
        # - ask if the user wants to explore a specific university
        # - use as initial prompt the data from that university and than use chatgpt

        # Display assistant response in chat message container
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = llama2.generate_llama2_response(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

