# AI localization project
Część projektu związana z obliczaniem rozkładu lokalizacji robota, była napisana na podstawie zadania z laboratorium nr 7.  
Rozszerzono strukturę dotyczącą lokalizacji, np. lokację *(6,9)* rozszerzono o kierunki, tak, aby mieć w jednej sturkturze wszystkie możliwe kierunki i współrzędne wszystkich lokacji.  
Uprościło to sprawę, gdyż, nie trzeba było tworzyć dodatkowej pętli for dla kierunków. Stworzono macierz tranzycji **T(168x168)**, macierz sensora **O(168x1)** oraz macierz z rozkładem **P(168x1)**.  
Funkcję **getPosterior** uzupełniono tak, aby do wymiaru **[loc,0]** były przypisane lokacje związane z kierunkiem N, do **[loc,1]** z kierunkiem E itd.

