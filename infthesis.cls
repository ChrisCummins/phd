%%
%%    File : infthesis.cls    (LaTeX2e class file)
%%  Author : Version 3.7 updated by Jennifer Oxley <joxley@inf.ed.ac.uk>
%%           Version 3.6 by Charles Sutton <csutton@inf.ed.ac.uk>  
%%           Version 3.5 by Vasilis Vasaitis <v.vasaitis@sms.ed.ac.uk>
%%           Version 3.4.1 updated by Tiejun Ma (t.j.ma@ed.ac.uk)
%%           Version 3.0 by Mary Ellen Foster <mef@cogsci.ed.ac.uk>
%%           Original version by Martin Reddy (mxr@dcs.ed.ac.uk) 
%% Version : 3.7
%% Updates : 1.0 [9/11/95] - initial release.
%%           1.1 [24/4/96] - fixed bibliography bug caused by new report.cls 
%%           1.2 [13/5/96] - \dedication & \thesiscaption[]
%%           1.3 [28/5/96] - abbrevs, parskip, minitoc fix, \headfootstyle
%%           1.4 [12/7/96] - appendices okay now, \cleardoublepage's added
%%           1.45 [6/8/96] - added space between chapter & numb on toc
%%           1.5 [13/8/96] - tailmargin was too small by 0.7cm!!
%%           2.0 [20/9/96] - \SetPrinter for margin settings (default=hp24)
%%                           no header, more abbreviations
%%           3.0 [16/10/2000] - Changed name and some formatting to become
%%                          "infthesis" instead
%%           3.1 [20/10/2000] - Added sans-serif running heads
%%                              Added pslatex by default (unless "cmfonts")
%%                              Added back in the code to create empty pages
%%                                  on cleardoublepage (from titlesec.sty)
%%           3.2 [13/11/2000] - Changed name of font-setting commands
%%                            - Changed name of shield input file
%%           3.3 [23/11/2000] - Use the new and improved shield graphic
%%           3.4 [23/11/2000] - Political changes... also fixed a problem
%%                              with the margins on two-sided documents.
%%           3.4.1[09/03/2006] -add [a4paper] parameter for geometry package
%%           3.5 [02/02/2011] - fix double-sided margins with modern
%%                              versions of the geometry package
%%           3.6 [15/03/2012] - addition to support the MInf degree
%%           3.7 [07/02/2013] - fix MInf definition


%% This file contains a class definition, infthesis, for the LaTeX2e
%% system which defines the layout of theses which are submitted in
%% the School of Informatics at the University of Edinburgh.
%%
%% For information on this class, please refer to "texdoc infthesis"
%%

\NeedsTeXFormat{LaTeX2e}[1994/12/01]
\ProvidesClass{infthesis}[2013/02/07 v3.7 School of Informatics Thesis Class]


%%
%% --- Initial Code ---
%%

%% Required packages:
%% - ifthen for conditionals
%% - geometry for margin-setting
%% - graphics for including the (scaled) logo on the front page
%% - xspace if abbreviations are used
\usepackage[paper=a4paper]{geometry}
\RequirePackage{ifthen,%  
    graphics,%
    xspace,
    eushield}


%%%% The following packages are also used but are included further on rather
%%%% than here, either because their options depend on the class options or
%%%% because they only need to be loaded in certain cases.
%% - parskip for alternate formatting of paragraphs
%% - tocbibind to put LOF and LOT into table of contents
%% - sectsty to change format of section headers
%% - caption to change format of captions
%% - pslatex to set default fonts (if they don't specify notimes)

%% Default values for various fields
\newcommand{\@degreetext}{}
\newcommand{\@infschool}{School of Informatics}
\newcommand{\@university}{University of Edinburgh}
\newcommand{\@chapteralignment}{\centering}
\newcommand{\@chapterfont}{}
\newcommand{\@thesisside}{}
\newcommand{\@thesisopen}{}
\newcommand{\@thesispoints}{}
\newcommand{\@draftmessage}{}

%% Lots of boolean things to keep track of options
\newboolean{draftthesis}
\newboolean{usequotemarks}
\newboolean{usesinglespacing}
\newboolean{usedoublespacing}
\newboolean{usefullspacing}
\newboolean{usedeptreport}
\newboolean{useabbrevs}
\newboolean{sansheadings}
\newboolean{ltoc}
\newboolean{romanpre}
\newboolean{logo}
\newboolean{frontabs}
\newboolean{strict}
\newboolean{timesfonts}

%% Choose the monochrome crest for the front page (if crests used)
\shieldtype{0}

%%
%%  --- Options ---
%%
%%  Current options: phd, mphil, msc, mscres, bsc
%%                   deptreport
%%                   draft
%%                   usequotes
%%                   singlespacing, doublespacing, fullspacing
%%                   cent{er,re}chapter, leftchapter, rightchapter,
%%                   + all report.cls options
%%

%% Commands to set course and project (for 4th year -- too many possibilities
%% to use options).
\let\@course\@empty
\newcommand{\course}[1]{\gdef\@course{#1}}
\newcommand{\@project}{Fourth Year Project Report}
\newcommand{\project}[1]{\gdef\@project{#1}}

%% Options to specify school and/or degree
% MSc degree: default is Informatics
\let\@mscdegree\@empty
\DeclareOption{cogsci}{\gdef\@mscdegree{Cognitive Science and Natural Language
Processing}}
\DeclareOption{cs}{\gdef\@mscdegree{Computer Science}}
\DeclareOption{ai}{\gdef\@mscdegree{Artificial Intelligence}}

% Institute
\let\@institute\@empty
\DeclareOption{aiai}{\gdef\@institute{Artificial Intelligence
Applications Institute}}
\DeclareOption{cisa}{\gdef\@institute{Centre for Intelligent Systems
and their Applications}}
\DeclareOption{icsa}{\gdef\@institute{Institute of Computing Systems
Architecture}}
\DeclareOption{ianc}{\gdef\@institute{Institute for Adaptive and
Neural Computation}}
\DeclareOption{ilcc}{\gdef\@institute{Institute for Language, Cognition and
Computation}}
\DeclareOption{ipab}{\gdef\@institute{Institute of Perception, Action
and Behaviour}}
\DeclareOption{lfcs}{\gdef\@institute{Laboratory for Foundations of
Computer Science}}

% Degree
\def\@researchdegree#1{%
    \renewcommand{\@degreetext}{#1 \\ 
    \ifx\@empty\@institute
        \PackageWarning{infthesis}{No institute specified for research
        degree}
    \else
        \@institute \\
    \fi
    \@infschool \\ \@university}
    \setboolean{strict}{true}}
\DeclareOption{phdproposal}{\@researchdegree{PhD Proposal}}
\DeclareOption{phd}{\@researchdegree{Doctor of Philosophy}}
\DeclareOption{mphil}{\@researchdegree{Master of Philosophy}}
\DeclareOption{mscres}{\@researchdegree{Master of Science by Research}}

\def\@taughtdegree#1#2{%
    \renewcommand{\@degreetext}{#1 \\ 
    \ifx\@empty#2
        % \PackageWarning{infthesis}{No course/school specified for taught
        % degree}
    \else
        #2 \\
    \fi
    \@infschool \\ \@university}%
    \setboolean{strict}{false}}
\DeclareOption{msc}{\@taughtdegree{Master of Science}{\@mscdegree}}
\DeclareOption{minf}{\@taughtdegree{Master of Informatics}{~}}
\DeclareOption{bsc}{\@taughtdegree{\@project}{\@course}}

%% Chapter header alignment, font of headings
\DeclareOption{centerchapter,centrechapter}
   {\renewcommand{\@chapteralignment}{\centering}}
\DeclareOption{leftchapter}
   {\renewcommand{\@chapteralignment}{\raggedright}}
\DeclareOption{rightchapter}
   {\renewcommand{\@chapteralignment}{\raggedleft}}
\DeclareOption{sansheadings}{%
    \setboolean{sansheadings}{true}}
\DeclareOption{normalheadings}{%
    \setboolean{sansheadings}{false}}

%% Sidedness, openright-ness, and font size (so that the draft option can
%% override them as needed. 
\DeclareOption{twoside}{\renewcommand{\@thesisside}{twoside}}
\DeclareOption{oneside}{\renewcommand{\@thesisside}{oneside}}
\DeclareOption{openany}{\renewcommand{\@thesisopen}{openany}}
\DeclareOption{openright}{\renewcommand{\@thesisopen}{openright}}
\DeclareOption{10pt}{\renewcommand{\@thesispoints}{10pt}}
\DeclareOption{11pt}{\renewcommand{\@thesispoints}{11pt}}
\DeclareOption{12pt}{\renewcommand{\@thesispoints}{12pt}}

%% Font options.
\DeclareOption{notimes}{\setboolean{timesfonts}{false}}
\DeclareOption{timesfonts}{\setboolean{timesfonts}{true}}

%% Whether it's a draft (if so, single space and wider margins to fit more)
\DeclareOption{draft}{
   \setboolean{draftthesis}{true}
   \ExecuteOptions{10pt,openany,twoside}
   \renewcommand{\@draftmessage}{(Draft Copy)}}

%% Use quotation marks in quotation environment?
\DeclareOption{usequotes}{\setboolean{usequotemarks}{true}}

%% Use useful abbrevations?
\DeclareOption{abbrevs}{\setboolean{useabbrevs}{true}}

%% Select spacing (default: fullspacing)
\DeclareOption{singlespacing}{\setboolean{usesinglespacing}{true}}
\DeclareOption{doublespacing}{\setboolean{usedoublespacing}{true}}
\DeclareOption{fullspacing}{\setboolean{usefullspacing}{true}}

%% Options to control the format of the cover page
\DeclareOption{deptreport}{\setboolean{usedeptreport}{true}}
\DeclareOption{logo}{%
    \setboolean{logo}{true}}
\DeclareOption{frontabs}{%
    \setboolean{frontabs}{true}}

%% Use parskip formatting of paragraphs
\DeclareOption{parskip}{\AtEndOfClass{\RequirePackage{parskip}}}

%% Whether to put list of figures and tables into TOC (default: no)
\DeclareOption{listsintoc}{%
    \setboolean{ltoc}{true}}

%% Pre pages can be numbered separately or with the rest of the thesis.
\DeclareOption{romanprepages}{\setboolean{romanpre}{true}}
\DeclareOption{plainprepages}{\setboolean{romanpre}{false}}

%% Pass all other options directly to report class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

%% Set default options and process the ones we were given.
\ExecuteOptions{phd,centerchapter,romanprepages,%
    sansheadings,openright,oneside,12pt,timesfonts} 
\ProcessOptions

%%
%% --- Class Loading (built ontop of report.cls) ---
%%

\LoadClass[a4paper,\@thesispoints,\@thesisside,\@thesisopen]{report}
\ifthenelse{\boolean{draftthesis}}
    {}
    {\ifthenelse{\boolean{strict}}
        {\if@twoside
            \if@openright
            \else
                \PackageWarning{infthesis}{A two-sided PhD or MPhil thesis must
                not use the "openany" option}
            \fi
        \fi}
        {}}

%%
%% --- Main Code ---
%%

\newboolean{isspecialchapter}
\setboolean{isspecialchapter}{false}

%%
%% First we will sort out the page layout. The following is a brief
%% summary of the university typesetting regulations:
%%   Printed on A4 paper, entirely on rectos (single-sided) or on both sides
%%      but with chapters starting on even pages
%%   4cm binding margin
%%   2cm head margin
%%   2.5cm fore-edge margin
%%   4cm tail margin
%%   spacing: not less then 1.5 spacing (18pt leading)
%%            quotations & footnotes in single spacing
%%            bibliography may be in single spacing
%%   character size: not exceed 12pt for body text (at least 10pt)
%%   style: a serif font should be used, with a sans-serif for headings
%%   hyphenation should be avoided if possible

%% Try to set up the margins according to the specifications in the thesis
%% regulations. I have removed all of the old code (ca. 1996) that attempted
%% to compensate for particular printers in the Department of Computer
%% Science.
\ifthenelse{\boolean{draftthesis}}
    {\geometry{a4paper,margin=2cm,twoside}}
    {\if@twoside
        \geometry{a4paper,left=4cm,top=2cm,right=2.5cm,bottom=4cm,twoside}
     \else
        \geometry{a4paper,left=4cm,top=2cm,right=2.5cm,bottom=4cm}
    \fi}

%% We should make pages created by "cleardoublepage" be
%% really empty. Taken from titlesec.sty

\def\cleardoublepage{%
    \clearpage{\ps@empty\if@twoside\ifodd\c@page\else
    \hbox{}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}}

%%
%% Hack to make minitoc work with csthesis. We declare a new chapter
%% variable called starchapter to be used by \addcontentsline when we
%% add contents lines for List of Figures/Tables. If we don't, then
%% minitoc treats the LOF/LOT sections as chapters of the thesis.
%%

\@ifundefined{chapter}{}{\let\l@starchapter\l@chapter}

%%
%% This bit will set up the header format for the thesis.
%% This currently uses a "headings" style showing the pagenumber
%% and chapter number/title. (in slanted text) If the document is two-sided,
%% the right-hand page will show the section number and title instead.
%%

\newcommand{\headfootstyle}{\normalsize}  % font size of headers and footers
% This will have \sffamily added to it if "sansheadings" is specified.

%% Set up different headers for right and left-hand pages. Those \defs are a
%% bit magic, but they seem to do the trick... :-)
%% Adapted from Francois Pitt's ut-thesis class from the University of
%% Toronto.
\if@twoside % if two-sided printing
\renewcommand{\ps@headings}{
   \let\@mkboth\markboth
   \def\@oddfoot{}
   \let\@evenfoot\@oddfoot
   \def\@oddhead{{\headfootstyle\itshape \rightmark}\hfil \headfootstyle\thepage}%
   \def\@evenhead{\headfootstyle\thepage\hfil
      {\headfootstyle\itshape\leftmark}}%
   \def\chaptermark##1{\markboth
      {\ifnum\c@secnumdepth >\m@ne
         \@chapapp\ \thechapter. \ \fi ##1}{}}%
   \def\sectionmark##1{\markright
      {\ifnum\c@secnumdepth >\z@
         \thesection. \ \fi ##1}}%
}%ps@headings
\else % if one-sided printing
\renewcommand{\ps@headings}{
   \let\@mkboth\markboth
   \def\@oddfoot{}
   \def\@oddhead{{\headfootstyle\itshape\rightmark}\hfil
      \headfootstyle\thepage}%
   \def\chaptermark##1{\markright
      {\ifnum\c@secnumdepth >\m@ne
         \@chapapp\ \thechapter. \ \fi ##1}}%
}%ps@headings
\fi%@twoside

\renewcommand{\ps@plain}{
  \renewcommand{\@oddfoot}{\hfil\headfootstyle\thepage\hfil}
  \renewcommand{\@evenfoot}{\hfil\headfootstyle\thepage\hfil}
  \renewcommand{\@evenhead}{}
  \renewcommand{\@oddhead}{}
}

%%
%% And now setup that headings style as default
%%

\newcommand{\@textpagenumbering}{arabic}
\newcommand{\@preamblepagenumbering}{roman}
\newcommand{\@textpagestyle}{headings}
\newcommand{\@preamblepagestyle}{plain}

\pagestyle{\@textpagestyle}

\setcounter{secnumdepth}{6}

%%
%% Set up the default names for the various chapter headings
%%

\renewcommand{\contentsname}{Table of Contents}
\renewcommand{\listfigurename}{List of Figures}
\renewcommand{\listtablename}{List of Tables}
\renewcommand{\bibname}{Bibliography}
\renewcommand{\indexname}{Index}
\renewcommand{\abstractname}{Abstract}

%%
%% Some sundry commands which are generally useful...
%%

\ifthenelse{\boolean{useabbrevs}}
 {\newcommand{\NB}{N.B.\@\xspace}
  \newcommand{\eg}{e.g.\@\xspace}
  \newcommand{\Eg}{E.g.\@\xspace}
  \newcommand{\ie}{i.e.\@\xspace}
  \newcommand{\Ie}{I.e.\@\xspace}
  \newcommand{\etc}{etc.\@\xspace}
  \newcommand{\etal}{{\em et al}.\@\xspace}
  \newcommand{\etseq}{{\em et seq}.\@\xspace}
  \newcommand{\precis}{pr\'ecis\xspace}
  \newcommand{\Precis}{Pr\'ecis\xspace}
  \newcommand{\role}{r\^ole\xspace}
  \newcommand{\Role}{R\^ole\xspace}
  \newcommand{\naive}{na\"\i ve\xspace}
  \newcommand{\Naive}{Na\"\i ve\xspace}
  \newcommand{\tm}{\raisebox{1ex}{\tiny TM}\xspace}
  \newcommand{\cpright}{\raisebox{1ex}{\tiny\copyright}\xspace}
  \newcommand{\degrees}{\raisebox{1.2ex}{\tiny\ensuremath{\circ}}\xspace}}
 {}

%%
%% Set up the double spacing and provide commands to alter the
%% spacing for the subsequent text. By default, 1.5 spacing will be
%% used. This can be modified through the singlespacing, doublespacing
%% or draft class options.
%%

\newcommand{\doublespace}{%
    \renewcommand{\baselinestretch}{1.66}\normalsize}
\newcommand{\oneandahalfspace}{%
    \renewcommand{\baselinestretch}{1.33}\normalsize}
\newcommand{\singlespace}{%
    \renewcommand{\baselinestretch}{1}\normalsize}

\ifthenelse{\boolean{draftthesis}}
   {\AtBeginDocument{\singlespace}}% \SetPrinterDraft}
   {\ifthenelse{\boolean{usesinglespacing}}
      {\AtBeginDocument{\singlespace}%
       \ifthenelse{\boolean{strict}}
        {\PackageWarning{infthesis}{Single spacing is not permitted in the
        regulations for PhD and MPhil theses}}
        {}}
      {\ifthenelse{\boolean{usedoublespacing}}
         {\AtBeginDocument{\doublespace}}
         {\AtBeginDocument{\oneandahalfspace}}
      }
   }

%%
%% We must ensure that the thesis ends on a lef-hand page. We
%% do the latter by issuing a \cleardoublepage at the end of document.
%% MEF: deleted this, there's no point -- it'll print the other side of that
%% page anyway!
%%

% \AtEndDocument{\cleardoublepage}

%%
%% A couple of commands for figures/captions
%%

\newcommand{\thesiscaption}[3][]{
   \ifthenelse{\equal{#1}{}}
     {\parbox{5in}{\caption{{\em #2\/}}\label{#3}}}
     {\parbox{5in}{\caption[#1]{{\em #2\/}}\label{#3}}}
}

%%
%% Quotations are supposed to be in single-space, so we will
%% explicitly redefine the quotation env. to support this.
%% And introduce a quotetext env. which can add an attribution.
%%

\let\old@quote\quote
\let\old@endquote\endquote
\renewcommand{\quote}{\old@quote\singlespace\ifthenelse{\boolean{usequotemarks}}{``}{}}
\renewcommand{\endquote}{\ifthenelse{\boolean{usequotemarks}}{''}{}\old@endquote}

\let\old@quotation\quotation
\let\old@endquotation\endquotation
\renewcommand{\quotation}{\old@quotation\singlespace}
\renewcommand{\endquotation}{\old@endquotation}
% \renewenvironment{quote}
%    {\old@quote\singlespace
%     \ifthenelse{\boolean{usequotemarks}}{``}{}}
%    {\ifthenelse{\boolean{usequotemarks}}{''}{}\end{quote}}

\newenvironment{iquote}
    {\begin{quote}\it}
    {\rm\end{quote}}

\newcommand{\quotationname}{}
\newenvironment{quotetext}[1]
   {\renewcommand{\quotationname}{#1}\begin{iquote}\singlespace
    \ifthenelse{\boolean{usequotemarks}}{``}{}\it}
   {\ifthenelse{\boolean{usequotemarks}}{\rm''}{}
    \hspace*{\fill}\nolinebreak[1]\hspace*{\fill}
    \rm (\quotationname)\end{iquote}}

%%
%% Footnotes should also be single-spaced.
%%

\let\tmp@footnotetext=\@footnotetext
\renewcommand{\@footnotetext}[1]%
    {{\singlespace\tmp@footnotetext{#1}}}

%% "preliminary" environment to control numbering of pages between title and
%% first chapter. This will only kick in if romanprepages is specificed (the
%% default).
%% Based on Francois Pitt's ut-thesis.cls from University of Toronto.

\newenvironment{preliminary}
    {\ifthenelse{\boolean{romanpre}}%
        {\pagestyle{plain}\pagenumbering{roman}}
        {\pagestyle{empty}}}%
    {\cleardoublepage%
     \ifthenelse{\boolean{romanpre}}%
        {\pagenumbering{arabic}}%
        {}}

%%
%% Let's have a dedication page so I can thank my mummy.
%%

\newcommand{\dedication}[1]
   {\vspace*{\fill}
    \begin{center}#1\end{center}
    \vspace*{\fill}}


%% A generic "frontmatter" environment, for use with abstract, dedication etc.
%% You specify the title of the environment and the font size to use (so that
%% both normal abstracts and those on the front page can be accommodated.)

\newenvironment{frontenv}[2]
  {\vspace{1cm}
   \begin{center}
       \@chapterfont\bfseries \LARGE#1
   \end{center}}
  {\par\vfil}

%% You specify the abstract with the \abs command; it gets automatically
%% inserted into the document where appropriate (title page or first main
%% page.

\def\@abs{}
\renewcommand{\abstract}[1]{\gdef\@abs{#1}}

\newenvironment{mainabs}
   {\begin{frontenv}{\abstractname}{\LARGE}}
   {\end{frontenv}\newpage}
\newenvironment{frontabs}
    {\begin{frontenv}{\abstractname}{\large}
     \begin{quotation}\rm}
    {\end{quotation}\end{frontenv}}

%%
%% Based upon the above abstract env., provide wrappers for
%% an acknowledgements and declation env.
%%

\newenvironment{acknowledgements}
   {\renewcommand{\abstractname}{Acknowledgements}\begin{mainabs}}
   {\end{mainabs}\renewcommand{\abstractname}{Abstract}}   

\newenvironment{declaration}
   {\renewcommand{\abstractname}{Declaration}\begin{mainabs}}
   {\end{mainabs}\renewcommand{\abstractname}{Abstract}}   

\newcommand{\standarddeclaration}{
   \begin{declaration}
   I declare that this thesis was composed by myself,
   that the work contained herein is my own 
   except where explicitly stated otherwise in the text,
   and that this work has not been submitted for any other degree or
   professional qualification except as specified.\par
   \vspace{1in}\raggedleft({\em \@author\/})
   \end{declaration}
}

%%
%% Now let's look at the format for the title page of the
%% thesis. This is done by redefining \maketitle, and allowing
%% some extra input options: \submityear and \graduationdate
%%

\def\submityear#1{\gdef\@submityear{#1}}
\gdef\@submityear{\the\year}

\def\graduationdate#1{\gdef\@graduationdate{#1}}
\gdef\@graduationdate{}

%% If usedeptreport is specified, then none of the other funky options kick
%% in. If not, then if frontabs is specified then it is used; otherwise,
%% the logo is inserted if required. If the abstract is not put on the front
%% page, then \maketitle will also create the abstract page as the first page
%% of the actual document.

\ifthenelse{\boolean{usedeptreport}}{
   \renewcommand{\maketitle}{
      \begin{titlepage}
      \addtolength{\oddsidemargin}{-0.75cm}
      \begin{center}
      \null\vskip 6.1cm
      \begin{minipage}[t][7.6cm]{10.5cm}
         \begin{center}
         {\LARGE\bfseries \@chapterfont \@title \par
            \ifthenelse{\boolean{draftthesis}}{\large \@draftmessage}{}
         }\vfil
         {\Large\itshape \@author \par}
         \end{center}
      \end{minipage}\\
      {\large \@degreetext \par \@submityear \par}
      \ifthenelse{\equal{\@graduationdate}{}}{}
         {\vskip 1cm {\large \ttfamily (Graduation date: \@graduationdate)}}
      \end{center}
      \end{titlepage}\cleardoublepage
      \begin{mainabs}\@abs\end{mainabs}
   }}{\ifthenelse{\boolean{frontabs}}{
     \ifthenelse{\boolean{strict}}
        {\PackageWarning{infthesis}{The regulations for PhD and MPhil theses
        do not permit the abstract on the front page}}
        {}
     \renewcommand{\maketitle}{
        \begin{titlepage}\begin{center}
        {\LARGE\bfseries \@chapterfont \@title \par
          \ifthenelse{\boolean{draftthesis}}{\large (Draft Copy)}{}
        }\vspace{3cm}
        {\Large\itshape \@author \par}\vspace{3cm}
        {\large \@degreetext \par \@submityear \par}
        \vskip 1cm 
        \ifthenelse{\equal{\@graduationdate}{}}{}
           {{\large \ttfamily (Graduation date: \@graduationdate)}}
        \end{center}
        \begin{frontabs}\@abs\end{frontabs}
        \end{titlepage}
     }}{
     \renewcommand{\maketitle}{
        \begin{titlepage}\begin{center}
        \null\vfil\vskip 60\p@
        {\LARGE\bfseries \@chapterfont \@title \par
           \ifthenelse{\boolean{draftthesis}}{\large (Draft Copy)}{}
        }\vfill
        {\Large\itshape \@author \par}\vskip 1cm\vfill
        \ifthenelse{\boolean{logo}}%
            {\resizebox{30mm}{!}{\includeshield}\\\vfill}
            {}
        {\large \@degreetext \par \@submityear \par}
        \vskip 1cm 
        \ifthenelse{\equal{\@graduationdate}{}}{}
           {{\large \ttfamily (Graduation date: \@graduationdate)}}
        \end{center}
        \end{titlepage}\cleardoublepage
        \begin{mainabs}\@abs\end{mainabs}
   }}}


%% If requested, put the list of figures and list of tables into the table of
%% contents.

\ifthenelse{\boolean{ltoc}}
    {\RequirePackage[nottoc,notbib]{tocbibind}}
    {}

%% Use the "pslatex" fonts unless they told us not to
\ifthenelse{\boolean{timesfonts}}
    {\RequirePackage{pslatex}}
    {}

%% ALWAYS put the bibliography into the TOC.
%% Thanks to Peter Wilson <peter.r.wilson@boeing.com> for pointing me in the
%% right direction, and to Heiko Oberdiek <oberdiek@ruf.uni-freiburg.de>
%% and Michael J Downes <epsmjd@ams,org> for together coming up with this
%% solution.
%%
%% However the bibliography is defined, this will append the \addcontentsline
%% statement to it. Also, it will put the bibname into the page headers.
\AtBeginDocument{%
    \expandafter\def\expandafter\thebibliography\expandafter
        #\expandafter1\expandafter{\thebibliography{#1}%
        {\addcontentsline{toc}{chapter}{\bibname}}
        \markboth{\bibname}{\bibname}}}

%% Do what is requested with headings and captions... can't include these
%% packages above because sectsty won't work except from within report.cls
%% itself. Must save \@chapterfont because the front environments (abstract
%% etc) also need to use it.
\RequirePackage{sectsty,caption}
\ifthenelse{\boolean{sansheadings}}
    {\allsectionsfont{\sffamily}
     \renewcommand{\@chapterfont}{\sffamily}
     \renewcommand{\captionfont}{\sffamily}
     \renewcommand{\headfootstyle}{\normalsize\sffamily}}
    {}
%% To make sure we get chapters aligned correctly, we set it here instead.
\chapterfont{\@chapterfont\@chapteralignment}

%%
%% EOF: infthesis.cls
%%
