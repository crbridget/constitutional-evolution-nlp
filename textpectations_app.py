

from great_textpectations import Textpectations
import textpectations_parsers as tp
import pprint as pp

def main():
    tt = Textpectations()
    tt.load_text('test.txt', 'A')

    pp.pprint(tt.data)




main()

