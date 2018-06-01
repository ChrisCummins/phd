%%
%% acaces.cls
%%
%% Michiel Ronsse <ronsse@elis.UGent.be>
%%
%% Loosely based on speauth.cls
%%
\def\update{2003/05/28 v0.1}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{acaces}[\update\ ACACES]
%%
\LoadClass[a4paper,12pt,twoside]{article}
\RequirePackage[latin1]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{palatino}
\RequirePackage[dvips]{graphicx}
\RequirePackage[usenames,dvipsnames]{color}
%\RequirePackage[pdftex]{hyperref}
\RequirePackage{hyperref}
\ProcessOptions\relax
%%

\def\pdfinfo#1{}

\frenchspacing
\flushbottom

\textheight 240mm
\topmargin 0mm

%odd:  25+170+15 = ([25] +  0) + 170 + 15
%even: 15+170+25 = ([25] - 10) + 170 + 25
\textwidth 170mm
\oddsidemargin 0mm
\evensidemargin -10mm



%%

\usepackage{fancyheadings}
\pagestyle{fancyplain}
\markboth{ACACES2004}{}

%%
\renewcommand{\footnoterule}{\noindent\rule{5pc}{0.25pt}\vspace*{6pt}}
\long\def\@makefntext#1{\parindent 1em%
\noindent{$\m@th^{\@thefnmark}$}#1}

\def\comma{$^{,}$}

\newcounter{affilcounter}
\def\addressnum#1{\setcounter{affilcounter}{#1}${}^{\fnsymbol{affilcounter}}$}

\def\extranum#1{\footnotemark[#1]}
\def\extra#1#2{\footnotetext[#1]{#2}\stepcounter{footnote}}

%%


\def\title#1{
\begingroup
\vspace*{5pt}%
\parbox{36pc}{\raggedright\huge{\bfseries\sffamily #1}}\\%
%\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\renewcommand{\thefootnote}{}
\vspace*{16pt}
\endgroup}

\def\author#1{\parbox{21pc}{\raggedright\large{#1}}\\[15pt]}
\def\address#1#2{\parbox{32pc}{\raggedright\small\textit{\setcounter{affilcounter}{#1}${}^{\fnsymbol{affilcounter}}$#2}}}


\def\abstract{
\vspace*{10pt}\noindent\small\textbf{%
\rule{\textwidth}{0.5pt}
\\[10pt]
ABSTRACT}\\[6pt]%
\small\begin{bfseries}\noindent\ignorespaces}%

\def\endabstract{\end{bfseries}\endlist\normalsize\par\noindent}

\def\keywords#1{\vspace{-3pt}\noindent{\small\textsc{keywords:}%
\hspace*{1em}#1}\vskip 0.1pt \normalsize\par\noindent}



\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}
\renewcommand{\@mkboth}[2]{}


% \bibliographystyle{alpha}
