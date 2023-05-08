
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fasttext
import pycountry

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "/Users/harshitkulkarni/Documents/language_detection/lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=2) # returns top 2 matching languages
        return predictions

if __name__ == '__main__':
    LANGUAGE = LanguageIdentification()
    lang = LANGUAGE.predict_lang("apa itu how are you doing , how are you doing , how are you doing , how are you doing")
    try:
        if any("__label__id" in label for label in lang):
            print("Bahasa")
        elif any('__label__en' in label for label in lang):
            print("English")
        else:
            print("English")
    except:
        print("no languae detected/ error on language detection")
