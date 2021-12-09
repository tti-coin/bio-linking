echo "Use umls; SELECT CUI, STR FROM MRCONSO WHERE SUPPRESS = 'N';" | mysql --user root -p --host 127.0.0.1 --port 3306 # > data/umls2017aa_full.sqlout
