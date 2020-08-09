import os
from twilio.rest import Client

def send_sms(x):
    account_sid= "*********" #from twilio account
    auth_token= "*************" #from twilio account
    client= Client(account_sid,auth_token)

    client.messages.create(
        to= "8988775659",
        from_="+1256777898", #provided by twilio
        body= f"Social distance violation count: {x}" #message
    )
    print("sms sent...")
