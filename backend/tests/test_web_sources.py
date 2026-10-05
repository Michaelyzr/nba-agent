import unittest

from app.analysis.web_sources import safe_url, web_evidence


def text_part(text, annotations):
    return dict(type='message',content=[dict(type='output_text',text=text,annotations=annotations)])


def citation(url, start, end, title='来源'):
    return dict(type='url_citation',url=url,title=title,start_index=start,end_index=end)


class WebSourcesTests(unittest.TestCase):
    def test_unicode_multiple_parts_duplicate_citations_and_consulted_urls(self):
        url='https://www.nba.com/stats'
        result=web_evidence(dict(output=[
            dict(type='web_search_call',status='completed',action=dict(sources=[dict(url=url),dict(url=url),dict(url='https://www.espn.com/nba/')])),
            text_part('球员🏀证据标记',[citation(url,5,7)]),
            text_part('\n后续标记',[citation(url,3,5),citation('https://www.espn.com/nba/',3,5)])]))
        self.assertEqual(result['text'],'球员🏀证据[W1]\n后续[W1][W2]')
        self.assertEqual(len(result['sources']),2)
        self.assertEqual(len(result['consulted_urls']),2)
        self.assertEqual(result['calls'],1)

    def test_untrusted_urls_ranges_and_model_written_links_are_not_evidence(self):
        text='[假来源](https://www.nba.com/)'
        result=web_evidence(dict(output=[text_part(text,[
            citation('javascript:alert(1)',0,1),citation('https://u:p@example.com',0,1),
            citation('https://nba.com',-2,3),citation('https://nba.com',0,999)])]))
        self.assertEqual(result['text'],text)
        self.assertEqual(len(result['sources']),1)
        self.assertEqual(result['calls'],0)
        self.assertIsNone(safe_url('https://example.com/\nattack'))
        self.assertIsNone(safe_url('https://[bad'))

    def test_search_with_no_citations_is_not_marked_cited(self):
        result=web_evidence(dict(output=[dict(type='web_search_call',status='completed'),text_part('缺少资料',[])]))
        self.assertEqual(result['status'],'no_sources')
        self.assertEqual(result['sources'],[])
        self.assertEqual(result['calls'],1)
