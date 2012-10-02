"""CMU phoneset."""

# Copyright 2011, 2012 Matt Shannon
# The follow copyrights may apply to the list of phone subsets:
#     Copyright 2001-2008 Nagoya Institute of Technology, Department of Computer Science
#     Copyright 2001-2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering
# The following copyrights may apply to the phoneset, which is based on the CMU pronouncing dictionary (which is in turn based on ARPAbet):
#     Copyright 1993-2008 Carnegie Mellon University

# This file is part of armspeech.
# See `License` for details of license and warranty.


from __future__ import division

class CmuPhoneset(object):
    def __init__(self):
        aa = 'aa'
        ae = 'ae'
        ah = 'ah'
        ao = 'ao'
        aw = 'aw'
        ax = 'ax'
        ay = 'ay'
        b = 'b'
        ch = 'ch'
        d = 'd'
        dh = 'dh'
        eh = 'eh'
        er = 'er'
        ey = 'ey'
        f = 'f'
        g = 'g'
        hh = 'hh'
        ih = 'ih'
        iy = 'iy'
        jh = 'jh'
        k = 'k'
        l = 'l'
        m = 'm'
        n = 'n'
        ng = 'ng'
        ow = 'ow'
        oy = 'oy'
        p = 'p'
        pau = 'pau'
        r = 'r'
        s = 's'
        sh = 'sh'
        t = 't'
        th = 'th'
        uh = 'uh'
        uw = 'uw'
        v = 'v'
        w = 'w'
        y = 'y'
        z = 'z'
        zh = 'zh'

        # extra phones not used in HTS demo version of CMU phoneset:
        axr = 'axr'
        brth = 'brth'
        dx = 'dx'
        el = 'el'
        em = 'em'
        en = 'en'
        hv = 'hv'
        h_hash = 'h#'
        ix = 'ix'
        nx = 'nx'

        self.phoneList = [
            aa,
            ae,
            ah,
            ao,
            aw,
            ax,
            ay,
            b,
            ch,
            d,
            dh,
            eh,
            er,
            ey,
            f,
            g,
            hh,
            ih,
            iy,
            jh,
            k,
            l,
            m,
            n,
            ng,
            ow,
            oy,
            p,
            pau,
            r,
            s,
            sh,
            t,
            th,
            uh,
            uw,
            v,
            w,
            y,
            z,
            zh
        ]

        vowelList = [aa,ae,ah,ao,aw,ax,axr,ay,eh,el,em,en,er,ey,ih,ix,iy,ow,oy,uh,uw]
        vowelSet = frozenset(vowelList)

        namedPhoneSubsetList = [
            ['Vowel', vowelList],
            ['Consonant', [b,ch,d,dh,dx,f,g,hh,hv,jh,k,l,m,n,nx,ng,p,r,s,sh,t,th,v,w,y,z,zh]],
            ['Stop', [b,d,dx,g,k,p,t]],
            ['Nasal', [m,n,en,ng]],
            ['Fricative', [ch,dh,f,hh,hv,s,sh,th,v,z,zh]],
            ['Liquid', [el,hh,l,r,w,y]],
            ['Front', [ae,b,eh,em,f,ih,ix,iy,m,p,v,w]],
            ['Central', [ah,ao,axr,d,dh,dx,el,en,er,l,n,r,s,t,th,z,zh]],
            ['Back', [aa,ax,ch,g,hh,jh,k,ng,ow,sh,uh,uw,y]],
            ['Front_Vowel', [ae,eh,ey,ih,iy]],
            ['Central_Vowel', [aa,ah,ao,axr,er]],
            ['Back_Vowel', [ax,ow,uh,uw]],
            ['Long_Vowel', [ao,aw,el,em,en,en,iy,ow,uw]],
            ['Short_Vowel', [aa,ah,ax,ay,eh,ey,ih,ix,oy,uh]],
            ['Dipthong_Vowel', [aw,axr,ay,el,em,en,er,ey,oy]],
            ['Front_Start_Vowel', [aw,axr,er,ey]],
            ['Fronting_Vowel', [ay,ey,oy]],
            ['High_Vowel', [ih,ix,iy,uh,uw]],
            ['Medium_Vowel', [ae,ah,ax,axr,eh,el,em,en,er,ey,ow]],
            ['Low_Vowel', [aa,ae,ah,ao,aw,ay,oy]],
            ['Rounded_Vowel', [ao,ow,oy,uh,uw,w]],
            ['Unrounded_Vowel', [aa,ae,ah,aw,ax,axr,ay,eh,el,em,en,er,ey,hh,ih,ix,iy,l,r,y]],
            ['Reduced_Vowel', [ax,axr,ix]],
            ['IVowel', [ih,ix,iy]],
            ['EVowel', [eh,ey]],
            ['AVowel', [aa,ae,aw,axr,ay,er]],
            ['OVowel', [ao,ow,oy]],
            ['UVowel', [ah,ax,el,em,en,uh,uw]],
            ['Unvoiced_Consonant', [ch,f,hh,k,p,s,sh,t,th]],
            ['Voiced_Consonant', [b,d,dh,dx,el,em,en,g,jh,l,m,n,ng,r,v,w,y]],
            ['Front_Consonant', [b,em,f,m,p,v,w]],
            ['Central_Consonant', [d,dh,dx,el,en,l,n,r,s,t,th,z,zh]],
            ['Back_Consonant', [ch,g,hh,jh,k,ng,sh,y]],
            ['Fortis_Consonant', [ch,f,k,p,s,sh,t,th]],
            ['Lenis_Consonant', [b,d,dh,g,jh,v,z,zh]],
            ['Neither_F_or_L', [el,em,en,hh,l,m,n,ng,r,w,y]],
            ['Coronal_Consonant', [ch,d,dh,dx,el,en,jh,l,n,r,s,sh,t,th,z,zh]],
            ['Non_Coronal', [b,em,f,g,hh,k,m,ng,p,v,w,y]],
            ['Anterior_Consonant', [b,d,dh,dx,el,em,en,f,l,m,n,p,s,t,th,v,w,z]],
            ['Non_Anterior', [ch,g,hh,jh,k,ng,r,sh,y,zh]],
            ['Continuent', [dh,el,em,en,f,hh,l,m,n,ng,r,s,sh,th,v,w,y,z,zh]],
            ['No_Continuent', [b,ch,d,g,jh,k,p,t]],
            ['Positive_Strident', [ch,jh,s,sh,z,zh]],
            ['Negative_Strident', [dh,f,hh,th,v]],
            ['Neutral_Strident', [b,d,el,em,en,g,k,l,m,n,ng,p,r,t,w,y]],
            ['Glide', [hh,l,el,r,y,w]],
            ['Syllabic_Consonant', [axr,el,em,en,er]],
            ['Voiced_Stop', [b,d,g]],
            ['Unvoiced_Stop', [p,t,k]],
            ['Front_Stop', [b,p]],
            ['Central_Stop', [d,t]],
            ['Back_Stop', [g,k]],
            ['Voiced_Fricative', [jh,dh,v,z,zh]],
            ['Unvoiced_Fricative', [ch,f,s,sh,th]],
            ['Front_Fricative', [f,v]],
            ['Central_Fricative', [dh,s,th,z]],
            ['Back_Fricative', [ch,jh,sh,zh]],
            ['Affricate_Consonant', [ch,jh]],
            ['Not_Affricate', [dh,f,s,sh,th,v,z,zh]],
            ['silence', [pau,h_hash,brth]],
            ['aa', [aa]],
            ['ae', [ae]],
            ['ah', [ah]],
            ['ao', [ao]],
            ['aw', [aw]],
            ['ax', [ax]],
            ['axr', [axr]],
            ['ay', [ay]],
            ['b', [b]],
            ['ch', [ch]],
            ['d', [d]],
            ['dh', [dh]],
            ['dx', [dx]],
            ['eh', [eh]],
            ['el', [el]],
            ['em', [em]],
            ['en', [en]],
            ['er', [er]],
            ['ey', [ey]],
            ['f', [f]],
            ['g', [g]],
            ['hh', [hh]],
            ['hv', [hv]],
            ['ih', [ih]],
            ['iy', [iy]],
            ['jh', [jh]],
            ['k', [k]],
            ['l', [l]],
            ['m', [m]],
            ['n', [n]],
            ['nx', [nx]],
            ['ng', [ng]],
            ['ow', [ow]],
            ['oy', [oy]],
            ['p', [p]],
            ['r', [r]],
            ['s', [s]],
            ['sh', [sh]],
            ['t', [t]],
            ['th', [th]],
            ['uh', [uh]],
            ['uw', [uw]],
            ['v', [v]],
            ['w', [w]],
            ['y', [y]],
            ['z', [z]],
            ['zh', [zh]],
            ['pau', [pau]],
            ['h#', [h_hash]],
            ['brth', [brth]]
        ]
        self.namedPhoneSubsets = [ (subsetName, frozenset(subsetList)) for subsetName, subsetList in namedPhoneSubsetList ]

        # slight hack to gather all named subsets relevant specifically to vowels
        self.namedVowelSubsets = [ (subsetName, subset) for subsetName, subset in self.namedPhoneSubsets if 'Vowel' in subsetName or subsetName in vowelSet ]
