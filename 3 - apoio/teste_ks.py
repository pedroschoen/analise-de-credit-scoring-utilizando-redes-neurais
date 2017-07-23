#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 19:49:48 2017

@author: pedroschoen
"""


    
def KS(x,verbose=True):
        
    import pandas
    import numpy

    x['mau']= 1 - x.CLASSE
    
    
    x['bucket'] = pandas.qcut(x.SCORE, 10)
    
    grouped = x.groupby('bucket', as_index = False)
    
    #numpy.savetxt("testeks.csv", x, fmt='%.2f',delimiter=";",)
    
    
    agg1 = grouped.min().SCORE
     
    agg1 = pandas.DataFrame(grouped.min().SCORE, columns = ['min_scr'])
     
    agg1['min_scr'] = grouped.min().SCORE
    
    
    agg1['max_scr'] = grouped.max().SCORE
     
    agg1['bads'] = grouped.sum().mau
    
    agg1['goods'] = grouped.sum().CLASSE
     
    agg1['total'] = agg1.bads + agg1.goods
    
     
    agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)
     
    agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
     
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
     
     
    agg2['ks'] = numpy.round(((agg2.bads / x.mau.sum()).cumsum() - (agg2.goods / x.CLASSE.sum()).cumsum()), 4) * 100
      
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
     
    agg2['max_ks'] = agg2.ks.apply(flag)
    
    if verbose:
        print ()
        print (agg2)

    return agg2.ks.max()
