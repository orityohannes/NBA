import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# step 1 - create dataframe
nbastats = {
  'Team': [
    'BOS', 'PHI', 'DAL', 'OKC', 'MIL', 'MIN', 'NYK', 'CLE', 'ATL', 'CHI',
    'POR', 'SAC', 'CHI', 'BOS', 'TOR', 'UTA', 'DEN', 'HOU', 'GSW', 'BKN',
    'GSW', 'NYK', 'DAL', 'MEM', 'LAL', 'NOP', 'MIA', 'ATL', 'SAC', 'GSW',
    'WAS', 'CLE', 'ORL', 'PHX', 'MIA', 'LAL', 'CHI', 'ORL', 'NYK', 'SAS',
    'BKN', 'PHX', 'WAS', 'MIA',
    'TOR', 'LAC', 'CHA', 'POR', 'IND', 'DEN', 'POR', 'MIL', 'CLE', 'DET',
    'UTA', 'LAL', 'OKC', 'MEM', 'MIL', 'LAC', 'SAC', 'PHI', 'PHI', 'NYK',
    'DET', 'PHX', 'TOR', 'MEM', 'WAS', 'IND', 'LAC', 'TOR', 'NOP', 'CHA',
    'SAC', 'HOU', 'TOR', 'NOP', 'DAL', 'IND', 'NOP', 'DEN', 'HOU', 'PHI',
    'DEN', 'ATL', 'OKC', 'MEM', 'SAC', 'HOU', 'ATL', 'LAL', 'LAC', 'DAL',
    'BOS', 'MIL', 'SAC', 'CHA', 'CLE', 'BKN', 'MIN', 'MIN', 'ATL', 'DEN',
    'MIA', 'CLE', 'TOR', 'SAS', 'ORL', 'LAC', 'LAC', 'UTA', 'CHA', 'ORL',
    'LAL', 'CHI', 'LAL', 'DEN', 'MEM', 'WAS', 'PHX', 'LAC', 'NYK', 'MIN',
    'ATL', 'POR', 'DET', 'ORL', 'MIN', 'PHI', 'ATL', 'HOU', 'ATL', 'PHI',
    'NOP', 'MIL', 'NYK', 'IND', 'CHA', 'SAS', 'LAC', 'CHI', 'LAC', 'TOR',
    'IND', 'LAL', 'SAC', 'MIA', 'MIA', "WAS", "MIN", "SAS", "BOS", "NOP",
    "UTA", "WAS", "MEM", "POR", "DEN", "OKC", "CHI", "UTA", "MIA", "UTA",
    "GSW", "SAS", "BKN", "CLE", "GSW", "LAC", "SAS", "NOP", "LAL", "DET",
    "MIL", "BKN", "IND", "NOP", "SAS", "MIN", "IND", "MIA", "BOS", "ATL",
    "PHI", "PHI", "WAS", "ORL", "GSW", "LAL", "GSW", "BOS", "MIA", "SAS",
    "DET", "BKN", "PHX", "MIN", "ORL", "PHX", "DEN", "GSW", "DET", "BKN",
    "SAC", "DAL", "BKN", "MEM", "BKN", "NOP", "DAL", "MEM", "IND", "LAL",
    "SAS", "CHA", "PHX", "ORL", "DET", "PHX", "POR", "OKC", "BOS", "MIA",
    "TOR", "DAL", "DET", "MIL", "NYK", "WAS", "PHX", "MIN", "CLE", "OKC",
    "LAC", "CHA", "UTA", "MEM", "MIL", "POR", "PHX", "IND", "SAC", "DET",
    "MIA", "NOP", "DEN", "NYK", "SAC", "MEM", "DET", "OKC", "DAL", "GSW",
    "POR", "CHI", "WAS", "NYK", "CHI", "IND", "ORL", "ATL", "POR", "CHA",
    "LAC", "BOS", "TOR", "BKN", "CHI", "HOU", "LAL", "WAS", "GSW", "MIL",
    "MEM", "MIN", "IND", "OKC", "MEM", "DEN", "POR", "DET", "PHX", "CLE",
    "UTA", "LAL", "BKN", "SAC", "PHI", "CHI", "MIL", "PHX", "GSW", "GSW",
    "DET", "SAS", "DEN", "CHA", "POR",'PHI'
  ],
  'Age': [
    25, 29, 24, 24, 28, 21, 28, 26, 24, 28, 32, 25, 33, 26, 29, 25, 28, 21, 23,
    26, 35, 26, 31, 23, 38, 31, 25, 26, 26, 33, 27, 23, 21, 26, 33, 30, 32, 20,
    22, 23, 30, 34, 27, 23, 30, 29, 32, 29, 23, 20, 26, 29, 32, 21, 34, 30, 27,
    20, 24, 35, 31, 30, 22, 33, 23, 21, 24, 21, 23, 29, 23, 34, 24, 22, 24, 24,
    22, 25, 30, 27, 27, 25, 27, 20, 30, 24, 24, 22, 27, 25, 22, 25, 26, 29, 31,
    28, 24, 19, 30, 28, 22, 27, 24, 24, 22, 30, 25, 26, 27, 28, 27, 23, 24, 33,
    34, 32, 21, 24, 24, 21, 29, 30, 26, 24, 37, 26, 22, 35, 22, 19, 21, 22, 23,
    24, 28, 21, 30, 25, 22, 27, 28, 23, 33, 25, 33, 23, 26, 30, 23, 25, 24, 23,
    22, 29, 25, 22, 22, 22, 28, 33, 23, 23, 24, 27, 21, 26, 19, 29, 28, 20, 22,
    31, 29, 24, 31, 27, 27, 31, 24, 27, 29, 23, 26, 24, 19, 29, 26, 27, 23, 28,
    24, 33, 36, 37, 19, 19, 21, 30, 27, 25, 32, 25, 27, 21, 31, 27, 32, 32, 26,
    29, 25, 22, 26, 24, 23, 28, 25, 24, 21, 24, 32, 26, 22, 25, 34, 23, 31, 24,
    31, 25, 27, 28, 29, 22, 24, 34, 25, 22, 22, 30, 22, 27, 21, 24, 22, 30, 30,
    36, 25, 25, 24, 31, 28, 20, 25, 23, 34, 24, 24, 29, 24, 28, 21, 23, 21, 32,
    31, 32, 27, 29, 21, 26, 30, 32, 36, 27, 24, 25, 29, 29, 22, 23, 24, 26, 25,
    27, 24, 28, 26, 29, 26, 35, 29, 25, 20, 22, 23, 26, 21, 26
],
  'GP': [
    74, 66, 66, 68, 63, 79, 77, 68, 73, 77, 58, 73, 74, 67, 71, 66, 69, 76, 82,
    83, 56, 68, 60, 61, 55, 75, 75, 74, 79, 69, 65, 69, 80, 53, 64, 56, 82, 72,
    73, 63, 79, 47, 64, 67, 80, 69, 56, 63, 62, 78, 65, 63, 67, 79, 59, 61, 71,
    76, 58, 78, 52, 82, 60, 58, 81, 74, 67, 77, 63, 50, 56, 73, 66, 79, 73, 75,
    59, 67, 79, 67, 62, 45, 68, 75, 74, 62, 77, 75, 73, 77, 82, 81, 60, 71, 82,
    74, 79, 67, 70, 80, 48, 68, 76, 79, 70, 71, 80, 80, 74, 72, 68, 57, 79, 69,
    68, 36, 60, 64, 82, 66, 76, 80, 74, 59, 76, 71, 67, 80, 80, 76, 60, 68, 77,
    65, 82, 54, 80, 29, 72, 76, 73, 50, 63, 65, 74, 81, 76, 75, 63, 63, 78, 65,
    38, 61, 77, 65, 76, 77, 52, 68, 73, 80, 48, 71, 74, 72, 66, 76, 77, 67, 56,
    64, 65, 56, 51, 81, 42, 75, 66, 67, 69, 68, 68, 79, 72, 78, 76, 62, 70, 37,
    78, 73, 63, 55, 56, 67, 57, 74, 29, 57, 79, 59, 82, 50, 74, 74, 78, 61, 56,
    66, 61, 60, 59, 62, 76, 73, 65, 72, 53, 56, 63, 78, 67, 80, 62, 55, 76, 42,
    33, 67, 70, 48, 54, 76, 70, 78, 54, 59, 70, 61, 62, 69, 63, 80, 45, 42, 65,
    56, 59, 64, 61, 62, 53, 48, 62, 63, 67, 62, 82, 67, 65, 48, 70, 40, 43, 34,
    63, 56, 53, 67, 64, 68, 50, 57, 58, 72, 59, 46, 57, 42, 76, 54, 52, 40, 62,
    52, 49, 58, 66, 57, 64, 46, 42, 45, 63, 47, 43, 60, 44, 71
  ],
  'W': [
    52, 43, 33, 33, 47, 40, 44, 44, 38, 38, 27, 44, 37, 46, 35, 32, 48, 20, 44,
    42, 30, 40, 32, 40, 30, 38, 40, 38, 47, 38, 27, 40, 33, 34, 35, 31, 40, 31,
    40, 17, 41, 34, 29, 37, 35, 34, 32, 19, 27, 33, 43, 28, 50, 48, 15, 30, 39,
    34, 38, 57, 33, 48, 39, 39, 47, 16, 36, 38, 41, 24, 28, 36, 31, 39, 24, 44,
    18, 33, 40, 33, 29, 23, 45, 18, 47, 41, 24, 36, 46, 47, 22, 35, 45, 31, 37,
    57, 36, 21, 46, 47, 23, 47, 16, 44, 41, 41, 34, 37, 53, 42, 45, 26, 20, 26,
    28, 25, 28, 13, 29, 35, 40, 37, 51, 51, 31, 33, 43, 41, 34, 41, 32, 16, 28,
    32, 50, 35, 22, 27, 31, 17, 53, 42, 30, 20, 20, 35, 37, 44, 39, 32, 32, 34,
    12, 43, 40, 30, 33, 47, 24, 36, 36, 40, 24, 38, 34, 42, 15, 42, 48, 38, 36,
    19, 24, 27, 12, 57, 25, 31, 31, 18, 36, 31, 37, 55, 34, 51, 52, 28, 26, 19,
    43, 41, 44, 30, 15, 16, 30, 40, 15, 27, 44, 33, 44, 12, 40, 43, 37, 37, 32,
    31, 30, 27, 37, 36, 40, 31, 21, 38, 20, 12, 27, 31, 31, 55, 35, 30, 37, 7,
    25, 37, 30, 27, 31, 48, 37, 44, 19, 29, 44, 43, 26, 35, 24, 47, 11, 23, 34,
    33, 35, 38, 40, 12, 26, 20, 36, 12, 34, 22, 47, 32, 27, 22, 35, 14, 18, 15,
    32, 27, 33, 33, 16, 37, 26, 30, 27, 45, 29, 19, 26, 28, 49, 22, 10, 19, 39,
    25, 25, 30, 41, 35, 29, 32, 26, 22, 37, 9, 14, 38, 14, 38
  ],
  'L': [
    22, 23, 33, 35, 16, 39, 33, 24, 35, 39, 31, 29, 37, 21, 36, 34, 21, 56, 38,
    41, 26, 28, 28, 21, 25, 37, 35, 36, 32, 31, 38, 29, 47, 19, 29, 25, 42, 41,
    33, 46, 38, 13, 35, 30, 45, 35, 24, 44, 35, 45, 22, 35, 17, 31, 44, 31, 32,
    42, 20, 21, 19, 34, 21, 19, 34, 58, 31, 39, 22, 26, 28, 37, 35, 40, 49, 31,
    41, 34, 39, 34, 33, 22, 23, 57, 27, 21, 53, 39, 27, 30, 60, 32, 36, 29, 34,
    25, 38, 58, 21, 23, 33, 32, 24, 35, 38, 36, 34, 27, 38, 29, 46, 48, 31, 51,
    44, 40, 23, 31, 29, 42, 29, 25, 29, 43, 26, 33, 30, 33, 39, 48, 60, 32, 36,
    27, 30, 60, 27, 49, 12, 19, 34, 43, 30, 43, 30, 37, 37, 37, 43, 25, 46, 31,
    26, 18, 37, 35, 43, 30, 28, 32, 37, 40, 24, 33, 40, 30, 51, 34, 29, 29, 20,
    45, 41, 29, 39, 24, 17, 44, 35, 49, 33, 37, 31, 24, 38, 27, 24, 34, 44, 18,
    35, 32, 19, 25, 41, 51, 27, 34, 14, 30, 35, 26, 38, 38, 34, 31, 41, 24, 24,
    35, 31, 33, 22, 26, 36, 42, 44, 34, 33, 44, 36, 47, 36, 25, 27, 25, 39, 35,
    8, 30, 40, 21, 23, 28, 33, 34, 35, 30, 26, 18, 36, 34, 39, 33, 34, 19, 31,
    23, 24, 26, 21, 50, 27, 28, 26, 51, 33, 40, 35, 35, 38, 26, 35, 26, 25, 19,
    31, 29, 20, 34, 48, 31, 24, 27, 31, 27, 30, 27, 31, 14, 27, 32, 42, 21, 23,
    27, 24, 28, 25, 22, 35, 14, 16, 23, 26, 38, 29, 22, 30, 33
  ],
}
players = [
  '1 Jayson Tatum ', '2 Joel Embiid', '3 Luka Doncic',
  '4 Shai Gilgeous-Alexander', '5 Giannis Antekounmpo', '6 Anthony Edwwards',
  '7 Julius Randle', '8 Donovan Mitchell', '9 Trae Young', '10 Zach LaVine',
  '11 Damian Lillard', '12 De Aaron Fox', '13 DeMar DeRozan',
  '14 Jaylen Brown', '15 Pascal Siakam', '16 Lauri Markkanen',
  '17 Nikola Jokic', '18 Jalen Green', '19 Jordan Poole', '20 Mikal Bridges',
  '21 Stephen Curry', '22 Jalen Brunson', '23 Kyrie Irving', '24 Ja Morant',
  '25 LeBron James', '26 CJ McCollum', '27 Bam Adebayo', '28 Dejounte Murray',
  '29 Domantas', '30 Klay Thompson', '31 Kristaps Porzingis',
  '32 Darius Garland', '33 Franz Wagner', '34 Devin Booker', '35 Jimmy Butler',
  '36 Anthony Davis', '37 Nikola Vuceviv', '38 Paola Banchero',
  '39 RJ Barrett', '40 Keldon Johnson', '41 Spencer Dinwiddle',
  '42 Kevin Durant', '43 Kyle Kuzma', '44 Tyler Herro', '45 Buddy Hield',
  '46 Fred VanVleet', '47 Paul George', '48 Terry Rozier',
  '49 Anfernee Simons', '50 Bennedict Mathurin', '51 Jamal Murray',
  '52 Jerami Grant', '52 Jrue Holiday', '54 Evan Mobley',
  '55 Bojan Bogdanivic', '56 Jordan Clarkson', '57 DAngelo Russell',
  '58 JOsh Giddey', '59 Desmond Russell', '60 Brook Lopez', '60 Kawhi Leonard',
  '62 Harrison Barnes', '63 Tyrese Maxey', '64 James Harden',
  '65 Immanuel Quickley', '66 Jaden Ivey', '67 Deandre Ayton',
  '68 Scottie Barnes', '69 Jaren Jackson Jr.', '70 Bradley Beal',
  '70 Tyresse Haluburton', '72 Russell Westbrook', '73 Gary Trent Jr.',
  '73 Trey Murphy III', '75 PJ Washington', '76 Kevin Huerter',
  '77 Kevin Porter Jr.', '78 O.G. Anunoby', '79 Jonas Valanciunas',
  '80 Christian Wood', '81 Myles Turner', '82 Brandon Ingram',
  '83 Aaron Gordon', '83 Alperen Sengun', '85 Tobias Harris',
  '86 Michael Porter Jr.', '87 Saddiq Bey', '88 Jalen Williams',
  '89 Jalen Williams', '89 Dillon Brooks', '90 Malik Monk',
  '91 Kenyon Martin Jr.', '92 DeAndre Hunter', '93 Malik Beasley',
  '94 Norman Powell', '94 Tim Hardaway Jr.', '96 Derrick White',
  '97 Luguentz Dort', '98 Jabari Smith Jr.', '99 Malcolm Brogdon',
  '100 Bobby Portis', '101 Keegan Murray', '102 Kelly Oubre Jr.',
  '103 Jarrett Allen', '104 Nic Claxton', '105 Jaden McDaniels',
  '106 Rudy Gobert', '107 John Collins', '108 Bruce Brown', '108 Max Strus',
  '110 Caris LeVert', '110 Jakon Poeltl', '112 Tre Jones',
  '113 Wendell Carter Jr.', '114 Mason Plumlee', '115 Eric Gordon',
  '116 Kelly Olynyk', '117 LaMelo Ball', '118 Markelle Fultz',
  '119 Austin Reaves', '119 Patrick Williams', '121 Dennis Schroder',
  '122 Kentavious Caldwell-Pope', '122 Tyus Jones', '124 Corey Kispert',
  '125 Chris Paul', '126 Ivica Zubac', '127 Quentin Grimes', '128 Mike Conley',
  '129 Onyeka Okongwu', '130 Shaedon Sharpe', '131 Killian Hayes',
  '132 Cole Anthony', '134 DeAnthony Melton', '135 Clint Capela',
  '136 Tari Eason', '137 Bogdan Bogdanovic', '138 Jalen McDaniels',
  '139 Zion Williamson', '140 Grayson Allen', '141 Josh Hart',
  '142 Aaron Nesmith', '143 Gordon Hayward', '144 Zach Collins',
  '145 Marcus Morris Sr.', '146 Coby White', '147 Terrance Mann',
  '148 Chris Boucher', '149 Andrew Nembhard', '150 Rui Hachimura',
  "151 Daniel Gafford", "152 Jaylen Nowell", "153 Devin Vassell",
  "155 Naji Marshall", "155 Talen Horton-Tucker", "157 Deni Avdija",
  "158 Santi Aldama", "159 Jusuf Nurkic", "160 Reggie Jackson",
  "161 Isaiah Joe", "162 Ayo Dosunmu", "163 Collin Sexton", "164 Caleb Martin",
  "165 Walker Kessler", "166 Donte DiVincenzo", "167 Malaki Branham",
  "168 Royce O'Neale", "169 Cedi Osman", "170 Jonathan Kuminga",
  "171 Bones Hyland", "172 Doug McDermott", "172 Josh Richardson",
  "172 Lonnie Walker IV", "175 Alec Burks", "176 Jevon Carter",
  "177 Cameron Johnson", "177 T.J. McConnell", "179 Herbert Jones",
  "180 Keita Bates-Diop", "180 Kyle Anderson", "182 Jalen Smith",
  "183 Gabe Vincent", "183 Grant Williams", "185 AJ Griffin",
  "186 Georges Niang", "186 Shake Milton", "188 Monte Morris", "189 Bol Bol",
  "190 Andrew Wiggins", "191 Jarred Vanderbilt", "192 Draymond Green",
  "193 Al Horford", "194 Kyle Lowry", "195 Jeremy Sochan", "196 Jalen Duren",
  "197 Cam Thomas", "197 Damion Lee", "199 Karl-Anthony Towns",
  "200 Moritz Wagner", "201 Torrey Craig", "202 Thomas Bryant",
  "203 Kevon Looney", "204 Isaiah Stewart", "205 Joe Harris", "205 Trey Lyles",
  "207 Reggie Bullock", "207 Seth Curry", "209 Brandon Clarke",
  "210 Dorian Finney-Smith", "211 Jose Alvarado", "212 Josh Green",
  "212 Luke Kennard", "214 Jordan Nwora", "215 Troy Brown Jr.",
  "216 Devonte' Graham", "217 Nick Richards", "218 Josh Okogie",
  "219 Jalen Suggs", "220 Hamidou Diallo", "220 Terrence Ross",
  "222 Drew Eubanks", "223 Tre Mann", "224 Sam Hauser", "225 Kevin Love",
  "226 Precious Achiuwa", "227 Dwight Powell", "228 Marvin Bagley III",
  "229 Khris Middleton", "229 Obi Toppin", "231 Kendrick Nunn",
  "232 Cameron Payne", "232 Taurean Prince", "234 Isaac Okoro",
  "235 Aaron Wiggins", "236 Nicolas Batum", "237 Dennis Smith Jr.",
  "238 Ochai Agbaji", "239 David Roddy", "239 Pat Connaughton",
  "241 Trendon Watford", "242 Jock Landale", "243 Isaiah Jackson",
  "244 Davion Mitchell", "244 James Wiseman", "246 Victor Oladipo",
  "247 Larry Nance Jr.", "248 Jeff Green", "249 Mitchell Robinson",
  "250 Terence Davis", "251 Xavier Tillman", "252 Cory Joseph",
  "253 Kenrich Williams", "254 Jaden Hardy", "255 Anthony Lamb",
  "255 Kevin Knox II", "255 Patrick Beverley", "258 Jordan Goodwin",
  "259 Isaiah Hartenstein", "260 Andre Drummond", "261 Oshae Brissett",
  "262 Gary Harris", "263 Jalen Johnson", "264 Cam Reddish",
  "264 Mark Williams", "266 John Wall", "267 Mike Muscala", "268 Will Barton",
  "269 Edmond Sumner", "270 Alex Caruso", "271 Josh Christopher",
  "271 Wenyen Gabriel", "271 Wenyen Gabriel", "273 Delon Wright",
  "274 JaMychal Green", "275 Goran Dragic", "276 John Konchar",
  "277 Nickeil Alexander-Walker", "278 Chris Duarte", "278 Dario Saric",
  "280 Steven Adams", "281 Christian Braun", "282 Nassir Little",
  "283 Isaiah Livers", "283 Landry Shamet", "285 Lamar Stevens",
  "285 Simone Fontecchio", "287 Mo Bamba", "288 Yuta Watanabe",
  "289 Chimezie Metu", "290 Montrezl Harrell", "291 Derrick Jones Jr.",
  "292 Joe Ingles", "293 T.J. Warren", "294 Ty Jerome", "295 Moses Moody",
  "295 R.J. Hampton", "297 Romeo Langford", "297 Vlatko Cancar",
  "299 Theo Maledon", "300 Matisse Thybulle"
]
df = pd.DataFrame(nbastats, index=players)
print(df.to_string())
print()

# step 2 clean & read data
df.to_csv('nbastats.csv', index=True)
df_save_file = pd.read_csv('nbastats.csv')
print(df_save_file)

print()
print(df.isnull())
print(df.duplicated())


# step 3 - stats
    # Age
mean_age = df['Age'].mean()
print("Mean of Age:",mean_age)

median_age = df['Age'].median()
print("Median of Age:", median_age)


var_age = df['Age'].var()
print('Variance of Age:', var_age)

std_age = df['Age'].std()
print('Standard deviation of Age:', std_age)
print()

    # W
mean_w = df['W'].mean()
print("Mean of W:",mean_w)

median_w = df['W'].median()
print("Median of W:", median_w)

var_w = df['W'].var()
print('Variance of W:', var_w)

std_w = df['W'].std()
print('Standard deviation of W:', std_w)
print()

# Step 4 

# 1
plt.figure(1)
plt.title('Top 10 Teams Charted')
y = np.array([10.256410256410255, 10.256410256410255, 10.256410256410255, 10.256410256410255, 10.256410256410255, 10.256410256410255, 10.256410256410255, 9.401709401709402, 9.401709401709402, 9.401709401709402])
mylabels = ['PHI (10.26%)', 'GSW (10.26%)', 'MEM (10.26%)', 'LAL (10.26%)', 'PHX (10.26%)', 'LAC (10.26%)', 'DET (10.26%)', 'POR (9.40%)', 'SAC (9.40%)', 'DEN (9.40%)']

plt.pie(y, labels = mylabels)
plt.legend(title = "Teams:")

# 2
plt.figure(2)
plt.title('Players Older/Younger Than the Average')
y = np.array([79, 21])
mylabels = ['<26', '>=26']
plt.pie(y, labels = mylabels)
plt.legend(title = "Key:")

# 3
plt.figure(3)
plt.title('Top 10 Players & Wins')
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([52, 43, 33, 33, 47, 40, 44, 44, 38, 38])

plt.scatter(x, y)
plt.xlabel('Player Ranking')
plt.ylabel('Amount of Wins')

# 4
plt.figure(4)
plt.title('Top 10 Players & Loses')
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([22, 23, 33, 35, 16, 39, 33, 24, 35, 39])

plt.scatter(x, y)
plt.xlabel('Player Ranking')
plt.ylabel('Amount of Loses')

plt.show()
