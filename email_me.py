import smtplib
from email.mime.text import MIMEText

def email_me(detail = '', title = ''):
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login('vesslaimsi@gmail.com', 'tboy jjhg tsak ffdw') #MSI12345678
    
    msg = MIMEText(detail)
    msg['Subject'] = title
    
    smtp.sendmail('vesslaimsi@gmail.com', 'jaeyoungi2006@gmail.com', msg.as_string()) #재영
    smtp.sendmail('vesslaimsi@gmail.com', 'mjk32384@gmail.com', msg.as_string()) #민준
    
    smtp.quit()