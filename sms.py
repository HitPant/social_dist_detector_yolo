import os
from twilio.rest import Client

def send_sms(x):
    account_sid= "ACd2e235572270ce48dc952b1dafff57bd" #from twilio account
    auth_token= "e13051f30d6641df7d90afcda7066fbb" #from twilio account
    client= Client(account_sid,auth_token)

    client.messages.create(
        to= "+919869016399",
        from_="+12058906507", #provided by twilio
        body= f"Social distance violation count: {x}. Name: Hemant.P.Sakhrani" #message
    )
    print("sms sent...")