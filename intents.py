import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


def predict_intent(user_input):
    model_path = 'issue_intent_classification_initial'
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    label_to_id={'shared_folder_access': 0,
    'new_hardware_request': 1,
    'vpn_setup': 2,
    'ticketing_system_guide': 3,
    'software_install': 4,
    'change_username': 5,
    'email_signature_setup': 6,
    'hardware_issue': 7,
    'ticket_status': 8,
    'reset_security_questions': 9,
    'access_issue': 10,
    'internet_issue': 12,
    'external_drive_issue': 13,
    'software_updates': 14,
    'slow_computer': 15,
    'change_language_settings': 16,
    'network_support': 17,
    'report_phishing': 18,
    'unlock_account': 19,
    'reset_password': 20}

    # Check if a GPU is available and use it; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model.to(device)

    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Map the predicted class ID to the label
    id_to_label = {i: label for label, i in label_to_id.items()}
    predicted_label = id_to_label[predicted_class_id]
    print("Classfied intent is "+ predicted_label)
    # final_response=get_response(predicted_label)
    return predicted_label
