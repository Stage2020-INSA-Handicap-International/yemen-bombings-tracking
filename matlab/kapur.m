function imagBW = kapur(imag)

imag = imag(:, :, 1); % gray image
counts = imhist(imag);  % counts are the histogram
GradeI = 256;    % the resolusion of the intensity. i.e. 256 for uint8.
% MIN = 1e-7;

prob = counts ./ sum(counts);  % Probability distribution

psai = zeros(GradeI, 1);    
prob_t = 0;
entropy_t = 0;              

ind = find(prob);
entropy_L = sum( prob(ind) .* log(prob(ind)) ) * (-1); 

for i = 0 : GradeI-1
    prob_t = prob_t + prob(i+1);
    
    if prob(i+1) > 0 && prob_t < 1
        entropy_t = entropy_t - prob(i+1) * log(prob(i+1));
        psai(i+1) = log(prob_t * (1-prob_t)) + entropy_t / prob_t + ...
            (entropy_L-entropy_t) / (1-prob_t);
    elseif (prob(i+1) == 0 && i > 0) 
        psai(i+1) = psai(i);
    end
end

[maxPsai, ind] = max(psai);
th = (ind - 1) / (GradeI -1);
imagBW = im2bw(imag, th);
th