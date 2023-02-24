using Plots
using StatsBase


function main()

    # fixed parameters
    nItems = 6
    humanStrict = [0.871429  0.710714  0.521429  0.342857  0.282143  0.546429]
    humanItem = [ 0.903571  0.8  0.685714  0.532143  0.432143  0.653571]

    # free parameters of the model
    Alpha = 0.4:0.1:0.9
    gamma = 0.15:0.15:0.9
    sigma = 0.01:0.01:0.06
    theta = 0.05:0.10:0.55
    nEstimParam = 6
    nTrials = nEstimParam*nEstimParam*nEstimParam*nEstimParam


    function primacy(Alpha, gamma, sigma, theta)
        # pre-allocate memory to store the to-be-recalled items
        recalled = zeros(nTrials,nItems)

        # Defines values to compute the score
        item = 1:6

        # item data results
        itemResult = zeros(nTrials,nItems)

        # strict data results
        strictResult = zeros(nTrials,nItems)

        # omissions data results
        omissionResult = zeros(nTrials,nItems)

        # strictScore
        strictScore = zeros(1,nItems)

        # itemScore
        itemScore = zeros(1,nItems)

        # omissionScore
        omissionScore = zeros(1,nItems)

        # gridSearchStrict
        gridSearchStrict = zeros(nTrials,nItems)

        # gridSearchItem
        gridSearchItem = zeros(nTrials,nItems)

        # repeat x times
        for epoch in 1:nTrials
            for i in nItems, j in nItems, k in nItems, l in nItems
                # creating the gradient
                gradient = Alpha[i] * gamma[j] .^ ((1:nItems) .- 1)

                # recall the items from the gradient
                for m in 1:nItems

                    # create a temporary gradient with noise
                    gradientNoise = gradient .+ (randn(nItems) * sigma[k])

                    # select the most activated one
                    selected = argmax(gradientNoise)

                    # place items or zeros for omissions
                    if gradientNoise[selected] <= theta[l]
                        recalled[epoch,m] = 0
                    else
                        # store that selected item
                        recalled[epoch,m] = copy(selected)
                    end

                    # then suppress its activation
                    gradient[selected] *= 0.05
                end

                # Compute the item score
                for n in 1:nItems
                    itemResult[epoch,:] += item[1:nItems] .== recalled[epoch,n]
                end
                itemResult = clamp.(itemResult, 0, 1)
                itemScore = mean(eachrow(itemResult))
                itemScore = reshape(itemScore, 1, 6)

                # Compute the strict score
                strictResult[epoch,:] = item .== recalled[epoch,:]
                strictScore = mean(eachrow(strictResult))
                strictScore = reshape(strictScore, 1, 6)

                # Compute the omissions score
                omissionResult[epoch,:] = 0 .== recalled[epoch,:]
                omissionScore = mean(eachrow(omissionResult))
                omissionScore = reshape(omissionScore, 1, 6)

            end

            # strore the result of the grid search
            gridSearchStrict[epoch,:] = strictScore
            gridSearchItem[epoch,:] = itemScore
        end

        # Compute the strict rmse
        strictDif = zeros(nTrials,nItems)
        strictDif = mean(eachcol(( gridSearchStrict .- humanStrict).^2))
        strictRmse = sqrt.(strictDif)
        indexStrict = argmin(strictRmse)

        # Retrieve the parameters for strict condition
        indexParamStrict = zeros(1,4)

        unIndex= indexStrict/216
        indexParamStrict[1,1]= ceil(unIndex)

        deuxIndex = (indexStrict-floor(unIndex)*216)/36
        indexParamStrict[1,2]= ceil(deuxIndex)

        troisIndex = (indexStrict-((floor(unIndex)*216) + (floor(deuxIndex)*36)))/6
        indexParamStrict[1,3]= ceil(troisIndex)

        quatreIndex = (indexStrict-((floor(unIndex)*216) + (floor(deuxIndex)*36) + (floor(troisIndex)*6)))/1
        indexParamStrict[1,4]= floor(quatreIndex)


        # Compute the item rmse
        itemDif = zeros(nTrials,nItems)
        itemDif = mean(eachcol(( gridSearchItem .- humanItem).^2))
        itemRmse = sqrt.(itemDif)
        indexItem = argmin(itemRmse)

        # Retrieve the parameters for item condition
        indexParamItem = zeros(1,4)

        unIndex= indexItem/216
        indexParamItem[1,1]= ceil(unIndex)

        deuxIndex = (indexItem-floor(unIndex)*216)/36
        indexParamItem[1,2]= ceil(deuxIndex)

        troisIndex = (indexItem-((floor(unIndex)*216) + (floor(deuxIndex)*36)))/6
        indexParamItem[1,3]= ceil(troisIndex)

        quatreIndex = (indexItem-((floor(unIndex)*216) + (floor(deuxIndex)*36) + (floor(troisIndex)*6)))/1
        indexParamItem[1,4]= floor(quatreIndex)


        score = vcat(strictScore, itemScore, omissionScore)
        return plot(1:nItems, transpose(score), xlabel = "Serial position", ylabel= "Recall performance", ylims=(0.0,1.0))

    end
end
primacy(Alpha, gamma, sigma, theta)
main()


