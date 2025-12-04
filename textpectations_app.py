
from great_textpectations import Textpectations
import textpectations_parsers as tp
import pprint as pp

def main():
    tt = Textpectations()
    tt.load_text('pdfs/usa_1787.pdf', 'USA (1797)', parser=tp.pdf_parser)
    tt.load_text('pdfs/france_1791.pdf', 'France (1791)', parser=tp.pdf_parser)
    tt.load_text('pdfs/mexico_1917.pdf', 'Mexico (1917)', parser=tp.pdf_parser)
    tt.load_text('pdfs/russia_1918.pdf', 'Russia 1918 (Soviet)', parser=tp.pdf_parser)
    tt.load_text('pdfs/germany_1919.pdf', 'Germany 1919 (Weimar)', parser=tp.pdf_parser)
    tt.load_text('pdfs/japan_1947.pdf', 'Japan 1947 (post-WWII)', parser=tp.pdf_parser)
    tt.load_text('pdfs/germany_1949.pdf', 'Germany 1949 (Basic Law)', parser=tp.pdf_parser)
    tt.load_text('pdfs/india_1950.pdf', 'India (1950)', parser=tp.pdf_parser)
    tt.load_text('pdfs/northKorea_1972.pdf', 'North Korea (1972)', parser=tp.pdf_parser)
    tt.load_text('pdfs/spain_1978.pdf', 'Spain (1978)', parser=tp.pdf_parser)
    tt.load_text('pdfs/iran_1979.pdf', 'Iran (1979)', parser=tp.pdf_parser)
    tt.load_text('pdfs/china_1982.pdf', 'China (1982)', parser=tp.pdf_parser)
    tt.load_text('pdfs/southKorea_1987.pdf', 'South Korea (1987)', parser=tp.pdf_parser)
    tt.load_text('pdfs/brazil_1988.pdf', 'Brazil (1988)', parser=tp.pdf_parser)
    tt.load_text('pdfs/russia_1993.pdf', 'Russia 1993 (post-USSR)', parser=tp.pdf_parser)
    tt.load_text('pdfs/southAfrica_1996.pdf', 'South Africa (1996)', parser=tp.pdf_parser)
    tt.load_text('pdfs/poland_1997.pdf', 'Poland (1997)', parser=tp.pdf_parser)

    tt.similarity_scatterplot()

    pp.pprint(tt.data)




main()

