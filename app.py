import PySimpleGUI as sg
import pandas as pd
from newsfetch.news import newspaper
from predict import predict_reliability
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import re


def is_reliable_news(url, model, tokenizer):
    def _is_valid(url):
        pattern = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(pattern, url) is not None

    if not _is_valid(url):
        raise ValueError(f'invalid URL: {url}')

    # Scrape news article
    news = newspaper(url)

    if news.language != 'en':
        raise NotImplementedError(f'Unsupported language: {news.language}')

    # Determine if news is reliable
    df = pd.DataFrame({'text': [news.headline + '\n\n' + news.article]})
    reliable = predict_reliability(model, tokenizer, df)

    return reliable


def main():
    sg.theme('Black')

    layout = [[sg.T('Pravda News Checker', font=("Helvetica", 16))],
              [sg.T('Check if news article is reliable for free')],
              [sg.T('URL'), sg.In(key='-INPUT-'), sg.Button('GO')],
              [sg.HSeparator()],
              [sg.T(size=(40, 1), key='-OUTPUT-')]]

    # Create the window
    window = sg.Window("Pravda", layout)

    # Load model
    model = RobertaForSequenceClassification.from_pretrained('saved-models/roBERTa-base-2/')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window
        if event == sg.WIN_CLOSED:
            break
        if event == 'GO':
            # window['-OUTPUT-'].update('Checking...')
            url = values['-INPUT-']
            try:
                if is_reliable_news(url, model, tokenizer):
                    window['-OUTPUT-'].update('Result: Reliable')
                else:
                    window['-OUTPUT-'].update('Result: Unreliable')
            except ValueError:
                window['-OUTPUT-'].update('Result: Error! Invalid URL')
            except NotImplementedError:
                window['-OUTPUT-'].update('Result: Error! The article must be in English')

    window.close()


if __name__ == '__main__':
    main()
