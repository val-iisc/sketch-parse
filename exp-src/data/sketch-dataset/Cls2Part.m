function mapping = Cls2Part(ClasName)

switch ClasName,
    case 'airplane'
        mapping.Parts = {'body', 'wheels', 'tail', 'engine', 'wings'};
        % this means that, body has label 0, wheels has 1, tail has 2, engine has 3 and wings has 4. 
        
    case 'cat'
        mapping.Parts = {'head',  'body', 'leg', 'tail'};

    case 'dog'
        mapping.Parts = {'head', 'body', 'leg', 'tail'};

    case 'cow'
        mapping.Parts = {'head',  'body', 'leg', 'tail'};

    case 'sheep'
        mapping.Parts = {'head', 'body',  'leg', 'tail'};

    case 'horse'
        mapping.Parts = {'head',  'body', 'leg', 'tail'};
        
    case 'bird'
        mapping.Parts = {'body','dont care','tail','dont care','dont care','head','leg','wing'};

    case 'bicycle'
        mapping.Parts = {'wheel','saddle','handlebar','body'};
        
    case 'bus'
        mapping.Parts = {'body','mirror','window','wheel','headlight','door'};

    case 'car'
        mapping.Parts = {'body','mirror','window','wheel','headlight','door'};
        
    case 'motorbike'
        mapping.Parts = {'wheel','saddle','handlebar','body'};
        
end

     
