"""Checks for the published workflow without reading study data."""
import copy
import unittest
from datetime import datetime,timedelta
import numpy as np
import modeling as m
from generate_synthetic_data import generate
from plotting import classify


class ModelingTests(unittest.TestCase):
    def setUp(self):self.events=generate()['events']

    def test_sample_matching_and_feature_expansion(self):
        event=self.events[0];rows=m.prepare_event(event)
        self.assertEqual(len(rows),37)
        truth={m.parse_time(p['timestamp']):p['value'] for p in event['cgm']}
        self.assertTrue(all(r['target']==truth[r['mapped_dt']] for r in rows))
        # E[x²] >= E[x]², and strict inequality in bins with varying signal.
        self.assertTrue(all(r['features'][2]+1e-12>=r['features'][1]**2 for r in rows))
        self.assertTrue(any(r['features'][2]>r['features'][1]**2+1e-6 for r in rows))

    def test_no_subject_pooling_or_heldout_fit(self):
        subjects,sources,errors=m.prepare_subjects(self.events)
        first=m.fit_subjects(subjects,sources,errors)
        self.assertEqual(len(first['coefficients']),3)
        self.assertEqual(len(first['points']),333)
        changed=copy.deepcopy(self.events)
        for e in changed:
            if e['event_id']!='event_1':
                for p in e['cgm']:p['value']+=7
        split,rows,errors=m.prepare_subjects(changed)
        second=m.fit_subjects(split,rows,errors)
        self.assertEqual(first['coefficients'],second['coefficients'])
        for subject in subjects:
            train=[p['timestamp'] for p in first['points'] if p['subject_id']==subject and p['split']=='train']
            test=[p['timestamp'] for p in first['points'] if p['subject_id']==subject and p['split']=='validation']
            self.assertTrue(set(train).isdisjoint(test))
            self.assertTrue(min(test)>max(train))

    def test_bad_earliest_event_not_replaced(self):
        events=copy.deepcopy(self.events[:3]);events[0]['seg'][0]['value']=float('nan')
        subjects,_,errors=m.prepare_subjects(events)
        self.assertFalse(subjects);self.assertTrue(errors)

    def test_minimum_calibration_samples(self):
        events=copy.deepcopy(self.events[:3])
        events[0]['cgm']=events[0]['cgm'][::6]  # Seven actual points; landmarks still covered.
        subjects,_,errors=m.prepare_subjects(events)
        self.assertFalse(subjects)
        self.assertTrue(any('need at least 12' in e['error'] for e in errors))

    def test_ridge_matches_standardized_closed_form(self):
        rows=m.prepare_event(self.events[0]);x=np.array([r['features'] for r in rows]);y=np.array([r['target'] for r in rows])
        means=x.mean(axis=0);means[0]=0
        scales=np.maximum(x.std(axis=0),1e-8);scales[0]=1
        z=(x-means)/scales;penalty=np.eye(6)*1e-6;penalty[0,0]=0
        expected=np.linalg.solve(z.T@z+penalty,z.T@y)/scales
        expected[0]-=np.sum(expected[1:]*means[1:])
        np.testing.assert_allclose(m.ridge_fit(rows),expected,rtol=1e-7,atol=1e-7)

    def test_overlap_and_duplicate_validation_samples(self):
        t=datetime(2035,1,1)
        make=lambda values:[{'mapped_dt':t+timedelta(minutes=i)} for i in values]
        events=[{'id':str(i),'subject_id':'S'} for i in range(3)]
        rows={'0':make(range(12)),'1':make(range(10,17)),'2':make(range(14,22))}
        split,_=m.chronological_split(events,rows,1)
        retained=[r['mapped_dt'] for group in split['S']['validation_rows'].values() for r in group]
        self.assertEqual(len(retained),10);self.assertEqual(len(set(retained)),10)
        self.assertEqual(split['S']['overlap_removed'],5)

    def test_clarke_boundaries(self):
        examples=[(100,100,'A'),(100,80,'A'),(100,120,'A'),(70,70,'A'),(50,60,'A'),
                  (100,140,'B'),(200,150,'B'),(100,250,'C'),(150,20,'C'),(50,100,'D'),
                  (300,120,'D'),(50,250,'E'),(250,50,'E')]
        self.assertEqual(classify([a for a,p,z in examples],[p for a,p,z in examples]).tolist(),[z for a,p,z in examples])


if __name__=='__main__':unittest.main()
